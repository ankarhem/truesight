use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use truesight_core::{
    Embedder, IncrementalStorage, IndexMetadata, IndexStats, IndexedCodeUnit, Language, Result,
    TruesightError,
};

use crate::parser::parse_file;
use crate::repo_context::{RepoContext, detect_repo_context};
use crate::util::hash_bytes;
use crate::walker::FileWalker;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ChangeSet {
    pub added: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
}

impl ChangeSet {
    fn normalize(&mut self) {
        self.added.sort();
        self.modified.sort();
        self.deleted.sort();
        self.added.dedup();
        self.modified.dedup();
        self.deleted.dedup();
    }

    fn total_files(&self) -> usize {
        self.added.len() + self.modified.len() + self.deleted.len()
    }
}

struct IndexAccumulators {
    languages: HashMap<Language, u32>,
    symbols_extracted: u32,
    chunks_embedded: u32,
}

struct IndexContext<'a, S: IncrementalStorage + ?Sized, E: Embedder + ?Sized> {
    root: &'a Path,
    storage: &'a S,
    embedder: &'a E,
    repo_id: &'a str,
    branch: &'a str,
}

#[derive(Debug, Clone, Default)]
pub struct IncrementalIndexer {
    walker: FileWalker,
}

impl IncrementalIndexer {
    pub fn new() -> Self {
        Self {
            walker: FileWalker::new(),
        }
    }

    pub async fn detect_changes<S: IncrementalStorage + ?Sized>(
        &self,
        root: &Path,
        storage: &S,
        repo_id: &str,
        branch: &str,
    ) -> Result<ChangeSet> {
        let metadata = storage.get_index_metadata(repo_id, branch).await?;
        if metadata.is_none() {
            return self.detect_initial_changes(root);
        }

        let metadata = metadata.expect("metadata checked above");
        if is_git_repo(root) {
            if let Some(last_sha) = metadata.last_commit_sha.as_deref() {
                if let Ok(changes) = git_changes(root, last_sha) {
                    return Ok(changes);
                }
            }
        }

        self.detect_hash_changes(root, storage, repo_id, branch)
            .await
    }

    pub async fn detect_changes_for_repo<S: IncrementalStorage + ?Sized>(
        &self,
        root: &Path,
        storage: &S,
    ) -> Result<(RepoContext, ChangeSet)> {
        let context = detect_repo_context(root)?;
        let changes = self
            .detect_changes(root, storage, &context.repo_id, &context.branch)
            .await?;
        Ok((context, changes))
    }

    pub async fn incremental_index<S: IncrementalStorage + ?Sized, E: Embedder + ?Sized>(
        &self,
        root: &Path,
        changes: &ChangeSet,
        storage: &S,
        embedder: &E,
        repo_id: &str,
        branch: &str,
    ) -> Result<IndexStats> {
        let started = Instant::now();
        let ctx = IndexContext {
            root,
            storage,
            embedder,
            repo_id,
            branch,
        };
        let mut accumulators = IndexAccumulators {
            languages: HashMap::new(),
            symbols_extracted: 0,
            chunks_embedded: 0,
        };

        for deleted in &changes.deleted {
            ctx.storage
                .delete_units_for_file(ctx.repo_id, ctx.branch, deleted)
                .await?;
            ctx.storage
                .delete_indexed_file(ctx.repo_id, ctx.branch, deleted)
                .await?;
        }

        for modified in &changes.modified {
            ctx.storage
                .delete_units_for_file(ctx.repo_id, ctx.branch, modified)
                .await?;
            self.index_one_file(&ctx, modified, &mut accumulators)
                .await?;
        }

        for added in &changes.added {
            self.index_one_file(&ctx, added, &mut accumulators).await?;
        }

        let all_symbols = ctx.storage.get_all_symbols(ctx.repo_id, ctx.branch).await?;
        let indexed_files = ctx
            .storage
            .get_indexed_files(ctx.repo_id, ctx.branch)
            .await?;
        let metadata = IndexMetadata {
            repo_id: ctx.repo_id.to_string(),
            branch: ctx.branch.to_string(),
            last_indexed_at: chrono::Utc::now(),
            last_commit_sha: current_commit_sha(ctx.root).ok(),
            file_count: indexed_files.len() as u32,
            symbol_count: all_symbols.len() as u32,
        };
        ctx.storage
            .set_index_metadata(ctx.repo_id, ctx.branch, &metadata)
            .await?;

        Ok(IndexStats {
            files_scanned: changes.total_files() as u32,
            files_indexed: (changes.added.len() + changes.modified.len()) as u32,
            files_skipped: 0,
            symbols_extracted: accumulators.symbols_extracted,
            chunks_embedded: accumulators.chunks_embedded,
            duration_ms: started.elapsed().as_millis() as u64,
            languages: accumulators.languages,
        })
    }

    pub async fn incremental_index_for_repo<
        S: IncrementalStorage + ?Sized,
        E: Embedder + ?Sized,
    >(
        &self,
        root: &Path,
        changes: &ChangeSet,
        storage: &S,
        embedder: &E,
    ) -> Result<(RepoContext, IndexStats)> {
        let context = detect_repo_context(root)?;
        let stats = self
            .incremental_index(
                root,
                changes,
                storage,
                embedder,
                &context.repo_id,
                &context.branch,
            )
            .await?;
        Ok((context, stats))
    }

    fn detect_initial_changes(&self, root: &Path) -> Result<ChangeSet> {
        let mut changes = ChangeSet {
            added: self
                .walker
                .walk(root)?
                .into_iter()
                .map(|file| relativize(root, &file.path))
                .collect::<Result<Vec<_>>>()?,
            modified: Vec::new(),
            deleted: Vec::new(),
        };
        changes.normalize();
        Ok(changes)
    }

    async fn detect_hash_changes<S: IncrementalStorage + ?Sized>(
        &self,
        root: &Path,
        storage: &S,
        repo_id: &str,
        branch: &str,
    ) -> Result<ChangeSet> {
        let indexed = storage.get_indexed_files(repo_id, branch).await?;
        let indexed_by_path = indexed
            .into_iter()
            .map(|record| (record.file_path, record.file_hash))
            .collect::<HashMap<_, _>>();
        let mut current_hashes = HashMap::new();

        for discovered in self.walker.walk(root)? {
            let relative_path = relativize(root, &discovered.path)?;
            let file_hash = hash_file(&discovered.path)?;
            current_hashes.insert(relative_path, file_hash);
        }

        let mut changes = ChangeSet::default();

        for (path, current_hash) in &current_hashes {
            match indexed_by_path.get(path) {
                None => changes.added.push(path.clone()),
                Some(stored_hash) if stored_hash != current_hash => {
                    changes.modified.push(path.clone())
                }
                Some(_) => {}
            }
        }

        let current_paths = current_hashes.keys().cloned().collect::<HashSet<_>>();
        for path in indexed_by_path.keys() {
            if !current_paths.contains(path) {
                changes.deleted.push(path.clone());
            }
        }

        changes.normalize();
        Ok(changes)
    }

    async fn index_one_file<S: IncrementalStorage + ?Sized, E: Embedder + ?Sized>(
        &self,
        ctx: &IndexContext<'_, S, E>,
        relative_path: &Path,
        accumulators: &mut IndexAccumulators,
    ) -> Result<()> {
        let absolute_path = ctx.root.join(relative_path);
        let language = Language::from_path(relative_path).ok_or_else(|| {
            TruesightError::Index(format!(
                "unsupported language for {}",
                relative_path.display()
            ))
        })?;
        let source = fs::read(&absolute_path)?;
        let file_hash = hash_bytes(&source);
        let units = parse_file(relative_path, &source, language)?;
        let mut stored_units = Vec::with_capacity(units.len());

        for unit in units {
            let embedding = ctx.embedder.embed(&unit.content)?;
            stored_units.push(IndexedCodeUnit {
                unit,
                embedding: Some(embedding),
                file_hash: Some(file_hash.clone()),
            });
        }

        ctx.storage
            .store_indexed_units(ctx.repo_id, ctx.branch, &stored_units)
            .await?;
        ctx.storage
            .upsert_indexed_file(ctx.repo_id, ctx.branch, relative_path, &file_hash)
            .await?;

        *accumulators.languages.entry(language).or_insert(0) += 1;
        accumulators.symbols_extracted += stored_units.len() as u32;
        accumulators.chunks_embedded += stored_units.len() as u32;
        Ok(())
    }
}

fn is_git_repo(root: &Path) -> bool {
    git_command(root, ["rev-parse", "--git-dir"]).is_ok()
}

fn current_commit_sha(root: &Path) -> Result<String> {
    git_command(root, ["rev-parse", "HEAD"])
}

fn git_changes(root: &Path, last_sha: &str) -> Result<ChangeSet> {
    let output = git_command(
        root,
        ["diff", "--name-status", "--no-renames", last_sha, "HEAD"],
    )?;
    let mut changes = ChangeSet::default();

    for line in output.lines().filter(|line| !line.trim().is_empty()) {
        let mut parts = line.split_whitespace();
        let status = parts.next().unwrap_or_default();
        let path = parts.next().ok_or_else(|| {
            TruesightError::Git(format!("missing path in git diff output line: {line}"))
        })?;
        let path = PathBuf::from(path);
        if Language::from_path(&path).is_none() {
            continue;
        }

        match status.chars().next().unwrap_or_default() {
            'A' => changes.added.push(path),
            'M' => changes.modified.push(path),
            'D' => changes.deleted.push(path),
            _ => {}
        }
    }

    changes.normalize();
    Ok(changes)
}

fn git_command<const N: usize>(root: &Path, args: [&str; N]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| TruesightError::Git(error.to_string()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(TruesightError::Git(if stderr.is_empty() {
            format!("git command failed: git {}", args.join(" "))
        } else {
            stderr
        }));
    }

    String::from_utf8(output.stdout)
        .map(|stdout| stdout.trim().to_string())
        .map_err(|error| TruesightError::Git(error.to_string()))
}

fn relativize(root: &Path, path: &Path) -> Result<PathBuf> {
    path.strip_prefix(root)
        .map(|relative| relative.to_path_buf())
        .map_err(|_| {
            TruesightError::Index(format!(
                "{} is outside repository root {}",
                path.display(),
                root.display()
            ))
        })
}

fn hash_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(hash_bytes(&bytes))
}

#[cfg(test)]
mod tests {
    use super::IncrementalIndexer;
    use crate::repo_context::detect_repo_context;
    use std::collections::HashSet;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use chrono::Utc;
    use tempfile::TempDir;
    use truesight_core::{
        Embedder, IncrementalStorage, IndexMetadata, IndexStorage, IndexedCodeUnit,
        IndexedFileRecord, Language, RankedResult, Storage,
    };

    #[derive(Clone, Default)]
    struct TestStorage {
        state: Arc<Mutex<StorageState>>,
    }

    #[derive(Default)]
    struct StorageState {
        units: Vec<IndexedCodeUnit>,
        indexed_files: Vec<IndexedFileRecord>,
        metadata: Option<IndexMetadata>,
    }

    #[async_trait]
    impl Storage for TestStorage {
        async fn store_code_units(
            &self,
            _repo_id: &str,
            _branch: &str,
            units: &[truesight_core::CodeUnit],
        ) -> truesight_core::Result<()> {
            self.state
                .lock()
                .unwrap()
                .units
                .extend(units.iter().cloned().map(|unit| IndexedCodeUnit {
                    unit,
                    embedding: None,
                    file_hash: None,
                }));
            Ok(())
        }

        async fn search_fts(
            &self,
            _repo_id: &str,
            _branch: &str,
            _query: &str,
            _limit: usize,
        ) -> truesight_core::Result<Vec<RankedResult>> {
            Ok(Vec::new())
        }

        async fn search_vector(
            &self,
            _repo_id: &str,
            _branch: &str,
            _embedding: &[f32],
            _limit: usize,
        ) -> truesight_core::Result<Vec<RankedResult>> {
            Ok(Vec::new())
        }

        async fn get_index_metadata(
            &self,
            _repo_id: &str,
            _branch: &str,
        ) -> truesight_core::Result<Option<IndexMetadata>> {
            Ok(self.state.lock().unwrap().metadata.clone())
        }

        async fn set_index_metadata(
            &self,
            _repo_id: &str,
            _branch: &str,
            meta: &IndexMetadata,
        ) -> truesight_core::Result<()> {
            self.state.lock().unwrap().metadata = Some(meta.clone());
            Ok(())
        }

        async fn delete_branch_index(
            &self,
            _repo_id: &str,
            _branch: &str,
        ) -> truesight_core::Result<()> {
            let mut state = self.state.lock().unwrap();
            state.units.clear();
            state.indexed_files.clear();
            state.metadata = None;
            Ok(())
        }

        async fn get_all_symbols(
            &self,
            _repo_id: &str,
            _branch: &str,
        ) -> truesight_core::Result<Vec<truesight_core::CodeUnit>> {
            Ok(self
                .state
                .lock()
                .unwrap()
                .units
                .iter()
                .map(|entry| entry.unit.clone())
                .collect())
        }
    }

    #[async_trait]
    impl IndexStorage for TestStorage {
        async fn store_indexed_units(
            &self,
            _repo_id: &str,
            _branch: &str,
            units: &[IndexedCodeUnit],
        ) -> truesight_core::Result<()> {
            self.state
                .lock()
                .unwrap()
                .units
                .extend(units.iter().cloned());
            Ok(())
        }

        async fn upsert_indexed_file(
            &self,
            _repo_id: &str,
            _branch: &str,
            file_path: &Path,
            file_hash: &str,
        ) -> truesight_core::Result<()> {
            let mut state = self.state.lock().unwrap();
            state
                .indexed_files
                .retain(|record| record.file_path != file_path);
            state.indexed_files.push(IndexedFileRecord {
                file_path: file_path.to_path_buf(),
                file_hash: file_hash.to_string(),
                indexed_at: Utc::now(),
            });
            state
                .indexed_files
                .sort_by(|left, right| left.file_path.cmp(&right.file_path));
            Ok(())
        }
    }

    #[async_trait]
    impl IncrementalStorage for TestStorage {
        async fn get_indexed_files(
            &self,
            _repo_id: &str,
            _branch: &str,
        ) -> truesight_core::Result<Vec<IndexedFileRecord>> {
            Ok(self.state.lock().unwrap().indexed_files.clone())
        }

        async fn delete_units_for_file(
            &self,
            _repo_id: &str,
            _branch: &str,
            file_path: &Path,
        ) -> truesight_core::Result<()> {
            self.state
                .lock()
                .unwrap()
                .units
                .retain(|entry| entry.unit.file_path != file_path);
            Ok(())
        }

        async fn delete_indexed_file(
            &self,
            _repo_id: &str,
            _branch: &str,
            file_path: &Path,
        ) -> truesight_core::Result<()> {
            self.state
                .lock()
                .unwrap()
                .indexed_files
                .retain(|record| record.file_path != file_path);
            Ok(())
        }
    }

    struct FakeEmbedder;

    impl Embedder for FakeEmbedder {
        fn embed(&self, text: &str) -> truesight_core::Result<Vec<f32>> {
            let mut vector = vec![0.0; 4];
            vector[0] = text.len() as f32;
            vector[1] = text.bytes().map(u32::from).sum::<u32>() as f32;
            vector[2] = text.lines().count() as f32;
            vector[3] = if text.contains("pub") { 1.0 } else { 0.0 };
            Ok(vector)
        }

        fn embed_batch(&self, texts: &[&str]) -> truesight_core::Result<Vec<Vec<f32>>> {
            texts.iter().map(|text| self.embed(text)).collect()
        }

        fn dimension(&self) -> usize {
            4
        }
    }

    #[tokio::test]
    async fn first_index_detects_all_supported_files_as_added() {
        let temp = git_repo();
        write_file(
            temp.path(),
            "src/lib.rs",
            "pub fn alpha() -> bool { true }\n",
        );
        write_file(
            temp.path(),
            "src/utils.rs",
            "pub fn beta() -> bool { false }\n",
        );
        git_commit_all(temp.path(), "initial");

        let database = TestStorage::default();
        let indexer = IncrementalIndexer::new();
        let context = detect_repo_context(temp.path()).expect("repo context should resolve");
        let changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .expect("initial detection should succeed");

        assert_eq!(
            change_paths(&changes.added),
            set_of(["src/lib.rs", "src/utils.rs"])
        );
        assert!(changes.modified.is_empty());
        assert!(changes.deleted.is_empty());
    }

    #[tokio::test]
    async fn git_detection_marks_only_modified_file_after_incremental_index() {
        let temp = git_repo();
        write_file(
            temp.path(),
            "src/lib.rs",
            "pub fn alpha() -> bool { true }\n",
        );
        write_file(
            temp.path(),
            "src/utils.rs",
            "pub fn beta() -> bool { false }\n",
        );
        git_commit_all(temp.path(), "initial");

        let database = TestStorage::default();
        let indexer = IncrementalIndexer::new();
        let embedder = FakeEmbedder;
        let context = detect_repo_context(temp.path()).expect("repo context should resolve");

        let initial_changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .unwrap();
        indexer
            .incremental_index(
                temp.path(),
                &initial_changes,
                &database,
                &embedder,
                &context.repo_id,
                &context.branch,
            )
            .await
            .unwrap();

        write_file(
            temp.path(),
            "src/lib.rs",
            "pub fn alpha_v2() -> bool { false }\n",
        );
        git_commit_all(temp.path(), "modify alpha");

        let changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .expect("git detection should succeed");

        assert_eq!(change_paths(&changes.modified), set_of(["src/lib.rs"]));
        assert!(changes.added.is_empty());
        assert!(changes.deleted.is_empty());
    }

    #[tokio::test]
    async fn incremental_index_replaces_units_for_modified_file_only() {
        let temp = git_repo();
        write_file(
            temp.path(),
            "src/lib.rs",
            "pub fn alpha() -> bool { true }\n",
        );
        write_file(
            temp.path(),
            "src/utils.rs",
            "pub fn beta() -> bool { false }\n",
        );
        git_commit_all(temp.path(), "initial");

        let database = TestStorage::default();
        let indexer = IncrementalIndexer::new();
        let embedder = FakeEmbedder;
        let context = detect_repo_context(temp.path()).expect("repo context should resolve");

        let initial_changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .unwrap();
        indexer
            .incremental_index(
                temp.path(),
                &initial_changes,
                &database,
                &embedder,
                &context.repo_id,
                &context.branch,
            )
            .await
            .unwrap();

        write_file(
            temp.path(),
            "src/lib.rs",
            "pub fn alpha_v2() -> bool { false }\n",
        );
        git_commit_all(temp.path(), "modify alpha");

        let changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .unwrap();
        let stats = indexer
            .incremental_index(
                temp.path(),
                &changes,
                &database,
                &embedder,
                &context.repo_id,
                &context.branch,
            )
            .await
            .unwrap();
        let stored = database
            .get_all_symbols(&context.repo_id, &context.branch)
            .await
            .unwrap();

        assert_eq!(stats.files_indexed, 1);
        assert_eq!(stats.files_scanned, 1);
        assert_eq!(stats.languages.get(&Language::Rust), Some(&1));
        assert_eq!(stored.len(), 2);
        assert!(stored.iter().any(|unit| unit.name == "alpha_v2"));
        assert!(stored.iter().any(|unit| unit.name == "beta"));
        assert!(!stored.iter().any(|unit| unit.name == "alpha"));
    }

    #[tokio::test]
    async fn incremental_index_cleans_deleted_files() {
        let temp = git_repo();
        write_file(
            temp.path(),
            "src/lib.rs",
            "pub fn alpha() -> bool { true }\n",
        );
        write_file(
            temp.path(),
            "src/utils.rs",
            "pub fn beta() -> bool { false }\n",
        );
        git_commit_all(temp.path(), "initial");

        let database = TestStorage::default();
        let indexer = IncrementalIndexer::new();
        let embedder = FakeEmbedder;
        let context = detect_repo_context(temp.path()).expect("repo context should resolve");

        let initial_changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .unwrap();
        indexer
            .incremental_index(
                temp.path(),
                &initial_changes,
                &database,
                &embedder,
                &context.repo_id,
                &context.branch,
            )
            .await
            .unwrap();

        fs::remove_file(temp.path().join("src/utils.rs")).unwrap();
        git_commit_all(temp.path(), "delete beta");

        let changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .unwrap();
        assert_eq!(change_paths(&changes.deleted), set_of(["src/utils.rs"]));

        indexer
            .incremental_index(
                temp.path(),
                &changes,
                &database,
                &embedder,
                &context.repo_id,
                &context.branch,
            )
            .await
            .unwrap();

        let stored = database
            .get_all_symbols(&context.repo_id, &context.branch)
            .await
            .unwrap();
        let indexed_files = database
            .get_indexed_files(&context.repo_id, &context.branch)
            .await
            .unwrap();

        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].name, "alpha");
        assert_eq!(indexed_files.len(), 1);
        assert_eq!(indexed_files[0].file_path, PathBuf::from("src/lib.rs"));
    }

    #[tokio::test]
    async fn non_git_detection_falls_back_to_indexed_file_hashes() {
        let temp = TempDir::new().unwrap();
        write_file(
            temp.path(),
            "src/lib.rs",
            "pub fn alpha() -> bool { true }\n",
        );
        write_file(
            temp.path(),
            "src/utils.rs",
            "pub fn beta() -> bool { false }\n",
        );

        let database = TestStorage::default();
        let indexer = IncrementalIndexer::new();
        let embedder = FakeEmbedder;
        let context = detect_repo_context(temp.path()).expect("repo context should resolve");

        let initial_changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .unwrap();
        indexer
            .incremental_index(
                temp.path(),
                &initial_changes,
                &database,
                &embedder,
                &context.repo_id,
                &context.branch,
            )
            .await
            .unwrap();

        write_file(
            temp.path(),
            "src/lib.rs",
            "pub fn alpha_changed() -> bool { false }\n",
        );
        fs::remove_file(temp.path().join("src/utils.rs")).unwrap();
        write_file(
            temp.path(),
            "src/new.rs",
            "pub fn gamma() -> bool { true }\n",
        );

        let changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .expect("hash fallback should succeed");

        assert_eq!(change_paths(&changes.added), set_of(["src/new.rs"]));
        assert_eq!(change_paths(&changes.modified), set_of(["src/lib.rs"]));
        assert_eq!(change_paths(&changes.deleted), set_of(["src/utils.rs"]));
    }

    fn change_paths(paths: &[PathBuf]) -> HashSet<String> {
        paths
            .iter()
            .map(|path| path.to_string_lossy().replace('\\', "/"))
            .collect()
    }

    fn set_of<const N: usize>(paths: [&str; N]) -> HashSet<String> {
        paths.into_iter().map(String::from).collect()
    }

    fn git_repo() -> TempDir {
        let temp = TempDir::new().unwrap();
        run_git(temp.path(), ["init", "-b", "main"]);
        temp
    }

    fn git_commit_all(root: &Path, message: &str) {
        run_git(root, ["add", "."]);
        let mut command = Command::new("git");
        command
            .args(["commit", "-m", message])
            .current_dir(root)
            .env("GIT_AUTHOR_NAME", "OpenCode")
            .env("GIT_AUTHOR_EMAIL", "opencode@example.com")
            .env("GIT_COMMITTER_NAME", "OpenCode")
            .env("GIT_COMMITTER_EMAIL", "opencode@example.com");
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "git commit failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn run_git<const N: usize>(root: &Path, args: [&str; N]) {
        let output = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn write_file(root: &Path, relative: &str, contents: &str) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }
}
