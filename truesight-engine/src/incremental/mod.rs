use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use truesight_core::{
    Embedder, IncrementalStorage, IndexMetadata, IndexStats, IndexStatus, Language, Result,
    TruesightError,
};

use crate::indexing::{build_pending_units, materialize_embeddings};
use crate::parser::parse_file;
use crate::repo_context::{RepoContext, detect_repo_context};
use crate::util::hash_bytes;
use crate::walker::FileWalker;

mod change_detection;
mod git;

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
            return change_detection::detect_initial_changes(&self.walker, root);
        }

        let metadata = metadata.expect("metadata checked above");
        if git::is_git_repo(root) {
            if let Some(last_sha) = metadata.last_commit_sha.as_deref() {
                if let Ok(changes) = git::git_changes(root, last_sha) {
                    return Ok(changes);
                }
            }
        }

        change_detection::detect_hash_changes(&self.walker, root, storage, repo_id, branch).await
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

        accumulators.chunks_embedded =
            materialize_embeddings(ctx.storage, ctx.repo_id, ctx.branch, ctx.embedder).await?;

        let all_symbols = ctx.storage.get_all_symbols(ctx.repo_id, ctx.branch).await?;
        let indexed_files = ctx
            .storage
            .get_indexed_files(ctx.repo_id, ctx.branch)
            .await?;
        let current_sha = git::current_commit_sha(ctx.root).ok();
        let metadata = IndexMetadata {
            repo_id: ctx.repo_id.to_string(),
            branch: ctx.branch.to_string(),
            status: IndexStatus::Ready,
            last_indexed_at: chrono::Utc::now(),
            last_commit_sha: current_sha.clone(),
            last_seen_commit_sha: current_sha,
            file_count: indexed_files.len() as u32,
            symbol_count: all_symbols.len() as u32,
            embedding_model: ctx.embedder.model_name().to_string(),
            last_error: None,
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
        let stored_units = build_pending_units(units, &file_hash);

        ctx.storage
            .store_indexed_units(ctx.repo_id, ctx.branch, &stored_units)
            .await?;
        ctx.storage
            .upsert_indexed_file(
                ctx.repo_id,
                ctx.branch,
                relative_path,
                &file_hash,
                stored_units.len() as u32,
            )
            .await?;

        *accumulators.languages.entry(language).or_insert(0) += 1;
        accumulators.symbols_extracted += stored_units.len() as u32;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{IncrementalIndexer, git};
    use crate::repo_context::detect_repo_context;
    use std::collections::HashSet;
    use std::fs;
    use std::path::{Path, PathBuf};

    use chrono::Utc;
    use tempfile::TempDir;
    use truesight_core::{
        IncrementalStorage, IndexMetadata, IndexStatus, IndexStorage, IndexedFileRecord, Language,
        MockIncrementalStorage, Storage,
    };

    mod storage_fakes {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/test_support/storage_fakes.rs"
        ));
    }

    use storage_fakes::{
        DeterministicFakeEmbedder as FakeEmbedder, RecordingStorage as TestStorage,
    };

    #[tokio::test]
    async fn generated_mock_incremental_storage_supports_full_trait_hierarchy() {
        let mut storage = MockIncrementalStorage::new();
        let repo_id = "/tmp/repo";
        let branch = "main";
        let indexed_file = IndexedFileRecord {
            file_path: PathBuf::from("src/lib.rs"),
            file_hash: "hash-alpha".to_string(),
            chunk_count: 1,
            indexed_at: Utc::now(),
        };
        let metadata = IndexMetadata {
            repo_id: repo_id.to_string(),
            branch: branch.to_string(),
            status: IndexStatus::Ready,
            last_indexed_at: Utc::now(),
            last_commit_sha: Some("abc123".to_string()),
            last_seen_commit_sha: Some("abc123".to_string()),
            file_count: 1,
            symbol_count: 1,
            embedding_model: "fake-model".to_string(),
            last_error: None,
        };

        storage
            .expect_search_fts()
            .times(1)
            .withf(|actual_repo, actual_branch, query, limit| {
                actual_repo == "/tmp/repo"
                    && actual_branch == "main"
                    && query == "alpha"
                    && *limit == 1
            })
            .returning(|_, _, _, _| Ok(Vec::new()));
        storage
            .expect_list_pending_embeddings()
            .times(1)
            .withf(|actual_repo, actual_branch, embedding_model, limit| {
                actual_repo == "/tmp/repo"
                    && actual_branch == "main"
                    && embedding_model == "fake-model"
                    && *limit == 4
            })
            .returning(|_, _, _, _| Ok(Vec::new()));
        storage
            .expect_get_indexed_files()
            .times(1)
            .withf(move |actual_repo, actual_branch| {
                actual_repo == repo_id && actual_branch == branch
            })
            .returning(move |_, _| Ok(vec![indexed_file.clone()]));
        storage
            .expect_delete_units_for_file()
            .times(1)
            .withf(|actual_repo, actual_branch, file_path| {
                actual_repo == "/tmp/repo"
                    && actual_branch == "main"
                    && file_path == Path::new("src/lib.rs")
            })
            .returning(|_, _, _| Ok(()));
        storage
            .expect_delete_indexed_file()
            .times(1)
            .withf(|actual_repo, actual_branch, file_path| {
                actual_repo == "/tmp/repo"
                    && actual_branch == "main"
                    && file_path == Path::new("src/lib.rs")
            })
            .returning(|_, _, _| Ok(()));
        storage
            .expect_set_index_metadata()
            .times(1)
            .withf(|actual_repo, actual_branch, actual_metadata| {
                actual_repo == "/tmp/repo"
                    && actual_branch == "main"
                    && actual_metadata.embedding_model == "fake-model"
                    && actual_metadata.status == IndexStatus::Ready
            })
            .returning(|_, _, _| Ok(()));

        let results = Storage::search_fts(&storage, repo_id, branch, "alpha", 1)
            .await
            .expect("generated MockIncrementalStorage should support Storage methods");
        assert!(results.is_empty());

        let pending =
            IndexStorage::list_pending_embeddings(&storage, repo_id, branch, "fake-model", 4)
                .await
                .expect("generated MockIncrementalStorage should support IndexStorage methods");
        assert!(pending.is_empty());

        let files = IncrementalStorage::get_indexed_files(&storage, repo_id, branch)
            .await
            .expect("generated MockIncrementalStorage should support IncrementalStorage methods");
        assert_eq!(files.len(), 1);

        IncrementalStorage::delete_units_for_file(
            &storage,
            repo_id,
            branch,
            Path::new("src/lib.rs"),
        )
        .await
        .expect("generated MockIncrementalStorage should delete units");
        IncrementalStorage::delete_indexed_file(&storage, repo_id, branch, Path::new("src/lib.rs"))
            .await
            .expect("generated MockIncrementalStorage should delete indexed files");
        Storage::set_index_metadata(&storage, repo_id, branch, &metadata)
            .await
            .expect("generated MockIncrementalStorage should support inherited storage updates");
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
        git::git_commit_all(temp.path(), "initial");

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
        git::git_commit_all(temp.path(), "initial");

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
        git::git_commit_all(temp.path(), "modify alpha");

        let changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .expect("git detection should succeed");

        assert_eq!(change_paths(&changes.modified), set_of(["src/lib.rs"]));
        assert!(changes.added.is_empty());
        assert!(changes.deleted.is_empty());
    }

    #[tokio::test]
    async fn git_detection_handles_paths_with_spaces_and_unicode() {
        let temp = git_repo();
        write_file(
            temp.path(),
            "src/naïve file.rs",
            "pub fn alpha() -> bool { true }\n",
        );
        write_file(
            temp.path(),
            "src/plain.rs",
            "pub fn beta() -> bool { false }\n",
        );
        git::git_commit_all(temp.path(), "initial");

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
            "src/naïve file.rs",
            "pub fn alpha_v2() -> bool { false }\n",
        );
        write_file(
            temp.path(),
            "src/plain.rs",
            "pub fn beta_v2() -> bool { true }\n",
        );
        git::git_commit_all(temp.path(), "modify unicode and spaced paths");

        let changes = indexer
            .detect_changes(temp.path(), &database, &context.repo_id, &context.branch)
            .await
            .expect("git detection should handle unicode and spaced paths");

        assert_eq!(
            change_paths(&changes.modified),
            set_of(["src/naïve file.rs", "src/plain.rs"])
        );
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
        git::git_commit_all(temp.path(), "initial");

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
        git::git_commit_all(temp.path(), "modify alpha");

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
        git::git_commit_all(temp.path(), "initial");

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
        git::git_commit_all(temp.path(), "delete beta");

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
        git::run_git(temp.path(), ["init", "-b", "main"]);
        temp
    }

    fn write_file(root: &Path, relative: &str, contents: &str) {
        let path = root.join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(path, contents).unwrap();
    }
}
