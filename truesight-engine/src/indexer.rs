use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::Utc;
use rayon::prelude::*;
use tracing::warn;
use truesight_core::{
    CodeUnit, Embedder, IndexMetadata, IndexStats, IndexStatus, IndexStorage, IndexedCodeUnit,
    IndexedFileRecord, Language, Result,
};

use crate::indexing::{build_pending_units, materialize_embeddings, repo_relative_path};
use crate::parser::CodeParser;
use crate::repo_context::detect_repo_context;
use crate::util::hash_bytes;
use crate::walker::{DiscoveredFile, FileWalker};

pub(crate) trait FileDiscovery: Send + Sync {
    fn walk(&self, root: &Path) -> Result<Vec<DiscoveredFile>>;
}

pub(crate) trait SourceParser: Send + Sync {
    fn parse_file(&self, path: &Path, source: &[u8], language: Language) -> Result<Vec<CodeUnit>>;
}

impl FileDiscovery for FileWalker {
    fn walk(&self, root: &Path) -> Result<Vec<DiscoveredFile>> {
        FileWalker::walk(self, root)
    }
}

impl SourceParser for CodeParser {
    fn parse_file(&self, path: &Path, source: &[u8], language: Language) -> Result<Vec<CodeUnit>> {
        CodeParser::parse_file(self, path, source, language)
    }
}

pub(crate) struct Indexer<W = FileWalker, P = CodeParser> {
    walker: W,
    parser: P,
}

impl Indexer<FileWalker, CodeParser> {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            walker: FileWalker::new(),
            parser: CodeParser::new()?,
        })
    }
}

impl<W, P> Indexer<W, P>
where
    W: FileDiscovery,
    P: SourceParser,
{
    #[cfg(test)]
    pub(crate) fn with_components(walker: W, parser: P) -> Self {
        Self { walker, parser }
    }

    pub(crate) async fn index_repo<S, E>(
        &self,
        root: &Path,
        storage: &S,
        embedder: &E,
    ) -> Result<IndexStats>
    where
        S: IndexStorage,
        E: Embedder,
    {
        let started_at = Instant::now();
        let context = detect_repo_context(root)?;
        let discovered = self.walker.walk(root)?;
        let files_scanned = discovered.len() as u32;

        let processed = discovered
            .par_iter()
            .enumerate()
            .filter_map(|(index, file)| {
                match process_file(index + 1, discovered.len(), root, file, &self.parser) {
                    Ok(outcome) => Some(outcome),
                    Err(error) => {
                        warn!(path = %file.path.display(), error = %error, "skipping file during indexing");
                        None
                    }
                }
            })
            .collect::<Vec<_>>();

        let mut indexed_units = Vec::new();
        let mut language_counts = HashMap::new();
        let mut symbols_extracted = 0_u32;

        for file in &processed {
            *language_counts.entry(file.language).or_insert(0) += 1;
            symbols_extracted += file.units.len() as u32;
            indexed_units.extend(file.units.iter().cloned());
        }
        let files_indexed = processed.len() as u32;

        let indexed_files = processed
            .iter()
            .map(|file| IndexedFileRecord {
                file_path: file.path.clone(),
                file_hash: file.file_hash.clone(),
                chunk_count: file.units.len() as u32,
                indexed_at: Utc::now(),
            })
            .collect::<Vec<_>>();

        let metadata = IndexMetadata {
            repo_id: context.repo_id.clone(),
            branch: context.branch.clone(),
            status: IndexStatus::Ready,
            last_indexed_at: Utc::now(),
            last_commit_sha: context.last_commit_sha.clone(),
            last_seen_commit_sha: context.last_commit_sha.clone(),
            file_count: files_indexed,
            symbol_count: symbols_extracted,
            embedding_model: embedder.model_name().to_string(),
            last_error: None,
        };

        storage
            .replace_branch_index(
                &context.repo_id,
                &context.branch,
                &indexed_units,
                &indexed_files,
                &metadata,
            )
            .await?;

        let chunks_embedded =
            materialize_embeddings(storage, &context.repo_id, &context.branch, embedder).await?;

        Ok(IndexStats {
            files_scanned,
            files_indexed,
            files_skipped: files_scanned.saturating_sub(files_indexed),
            symbols_extracted,
            chunks_embedded,
            duration_ms: started_at.elapsed().as_millis() as u64,
            languages: language_counts,
        })
    }
}

pub async fn index_repo<S, E>(root: &Path, storage: &S, embedder: &E) -> Result<IndexStats>
where
    S: IndexStorage,
    E: Embedder,
{
    Indexer::new()?.index_repo(root, storage, embedder).await
}

fn process_file<P>(
    index: usize,
    total: usize,
    root: &Path,
    file: &DiscoveredFile,
    parser: &P,
) -> Result<ProcessedFile>
where
    P: SourceParser,
{
    tracing::info!(file_number = index, total_files = total, path = %file.path.display(), "indexing file");

    let source = fs::read(&file.path)?;
    let file_hash = hash_bytes(&source);
    let relative_path = repo_relative_path(root, &file.path)?;
    let parsed_units = parser.parse_file(&relative_path, &source, file.language)?;

    if parsed_units.is_empty() {
        return Ok(ProcessedFile {
            path: relative_path,
            language: file.language,
            file_hash,
            units: Vec::new(),
        });
    }

    let units = build_pending_units(parsed_units, &file_hash);

    Ok(ProcessedFile {
        path: relative_path,
        language: file.language,
        file_hash,
        units,
    })
}

#[derive(Debug, Clone)]
struct ProcessedFile {
    path: PathBuf,
    language: Language,
    file_hash: String,
    units: Vec<IndexedCodeUnit>,
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    mod git_fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/test_support/git_fixture.rs"
        ));
    }

    mod storage_fakes {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/test_support/storage_fakes.rs"
        ));
    }

    use tempfile::TempDir;
    use truesight_core::{
        IndexStatus, IndexStorage, IndexedCodeUnit, IndexedFileRecord, MockIndexStorage,
        TruesightError,
    };

    use super::*;
    use git_fixture::{TempGitFixture, init_empty_git_repo};
    use storage_fakes::{DeterministicFakeEmbedder as FakeEmbedder, RecordingStorage};

    struct ControlledParser {
        inner: CodeParser,
    }

    impl ControlledParser {
        fn new() -> Self {
            Self {
                inner: CodeParser::new().expect("parser should initialize"),
            }
        }
    }

    impl SourceParser for ControlledParser {
        fn parse_file(
            &self,
            path: &Path,
            source: &[u8],
            language: Language,
        ) -> Result<Vec<CodeUnit>> {
            if path.file_name().and_then(|name| name.to_str()) == Some("broken.rs") {
                return Err(TruesightError::Parse(
                    "synthetic parser failure".to_string(),
                ));
            }
            self.inner.parse_file(path, source, language)
        }
    }

    #[tokio::test]
    async fn generated_mock_index_storage_supports_replace_branch_index() {
        let mut storage = MockIndexStorage::new();
        let repo_id = "/tmp/repo";
        let branch = "main";
        let units = vec![IndexedCodeUnit {
            unit: CodeUnit {
                name: "alpha".to_string(),
                kind: truesight_core::CodeUnitKind::Function,
                signature: "pub fn alpha()".to_string(),
                doc: None,
                file_path: PathBuf::from("src/lib.rs"),
                line_start: 1,
                line_end: 3,
                content: "pub fn alpha() {}".to_string(),
                parent: None,
                language: Language::Rust,
            },
            embedding: None,
            file_hash: Some("hash-alpha".to_string()),
        }];
        let indexed_files = vec![IndexedFileRecord {
            file_path: PathBuf::from("src/lib.rs"),
            file_hash: "hash-alpha".to_string(),
            chunk_count: 1,
            indexed_at: Utc::now(),
        }];
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
            .expect_delete_branch_index()
            .times(1)
            .withf(move |actual_repo, actual_branch| {
                actual_repo == repo_id && actual_branch == branch
            })
            .returning(|_, _| Ok(()));
        storage
            .expect_store_indexed_units()
            .times(1)
            .withf(|actual_repo, actual_branch, actual_units| {
                actual_repo == "/tmp/repo"
                    && actual_branch == "main"
                    && actual_units.len() == 1
                    && actual_units[0].unit.name == "alpha"
            })
            .returning(|_, _, _| Ok(()));
        storage
            .expect_upsert_indexed_file()
            .times(1)
            .withf(
                |actual_repo, actual_branch, file_path, file_hash, chunk_count| {
                    actual_repo == "/tmp/repo"
                        && actual_branch == "main"
                        && file_path == Path::new("src/lib.rs")
                        && file_hash == "hash-alpha"
                        && *chunk_count == 1
                },
            )
            .returning(|_, _, _, _, _| Ok(()));
        storage
            .expect_set_index_metadata()
            .times(1)
            .withf(|actual_repo, actual_branch, actual_metadata| {
                actual_repo == "/tmp/repo"
                    && actual_branch == "main"
                    && actual_metadata.embedding_model == "fake-model"
                    && actual_metadata.symbol_count == 1
            })
            .returning(|_, _, _| Ok(()));

        IndexStorage::replace_branch_index(
            &storage,
            repo_id,
            branch,
            &units,
            &indexed_files,
            &metadata,
        )
        .await
        .expect("generated MockIndexStorage should support replace_branch_index");
    }

    #[tokio::test]
    async fn index_rust_fixture_persists_expected_symbols_and_stats() {
        assert_fixture_index("rust-fixture", 2, Language::Rust).await;
    }

    #[tokio::test]
    async fn index_typescript_fixture_persists_expected_symbols_and_stats() {
        assert_fixture_index("ts-fixture", 2, Language::TypeScript).await;
    }

    #[tokio::test]
    async fn index_csharp_fixture_persists_expected_symbols_and_stats() {
        assert_fixture_index("csharp-fixture", 3, Language::CSharp).await;
    }

    #[tokio::test]
    async fn index_repo_handles_empty_repository() {
        let temp_dir = TempDir::new().expect("temp dir should exist");
        fs::write(temp_dir.path().join("README.md"), "empty\n")
            .expect("readme write should succeed");

        let storage = RecordingStorage::default();
        let stats = index_repo(temp_dir.path(), &storage, &FakeEmbedder)
            .await
            .expect("empty repo should index successfully");

        assert_eq!(stats.files_scanned, 0);
        assert_eq!(stats.files_indexed, 0);
        assert_eq!(stats.files_skipped, 0);
        assert_eq!(stats.symbols_extracted, 0);
        assert_eq!(stats.chunks_embedded, 0);
        assert!(stats.languages.is_empty());

        let state = storage.state.lock().expect("storage lock poisoned");
        assert!(state.stored_units.is_empty());
        assert!(state.indexed_files.is_empty());
        let metadata = state.metadata.clone().expect("metadata should be recorded");
        assert_eq!(metadata.file_count, 0);
        assert_eq!(metadata.symbol_count, 0);
    }

    #[tokio::test]
    async fn parse_failure_in_one_file_does_not_abort_other_files() {
        let temp_dir = git_repo_with_sources();
        fs::write(
            temp_dir.path().join("good.rs"),
            "/// healthy\npub fn healthy() -> bool { true }\n",
        )
        .expect("good source write should succeed");
        fs::write(
            temp_dir.path().join("broken.rs"),
            "/// broken\npub fn broken() -> bool { true }\n",
        )
        .expect("broken source write should succeed");

        let storage = RecordingStorage::default();
        let indexer = Indexer::with_components(FileWalker::new(), ControlledParser::new());
        let stats = indexer
            .index_repo(temp_dir.path(), &storage, &FakeEmbedder)
            .await
            .expect("indexing should continue after one parser failure");

        assert_eq!(stats.files_scanned, 2);
        assert_eq!(stats.files_indexed, 1);
        assert_eq!(stats.files_skipped, 1);
        assert_eq!(stats.symbols_extracted, 1);
        assert_eq!(stats.chunks_embedded, 1);

        let stored = storage
            .state
            .lock()
            .expect("storage lock poisoned")
            .stored_units
            .iter()
            .map(|entry| entry.unit.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(stored, vec!["healthy".to_string()]);
        let metadata = storage
            .state
            .lock()
            .expect("storage lock poisoned")
            .metadata
            .clone()
            .expect("metadata should be recorded");
        assert_eq!(metadata.file_count, 1);
        assert_eq!(metadata.symbol_count, 1);
    }

    #[tokio::test]
    async fn index_repo_tracks_supported_files_without_public_symbols() {
        let temp_dir = TempDir::new().expect("temp dir should exist");
        init_empty_git_repo(temp_dir.path());
        fs::create_dir_all(temp_dir.path().join("src")).expect("src dir should exist");
        fs::write(
            temp_dir.path().join("src/internal.rs"),
            "fn hidden_example() -> bool { true }\n",
        )
        .expect("source write should succeed");

        let storage = RecordingStorage::default();
        let stats = index_repo(temp_dir.path(), &storage, &FakeEmbedder)
            .await
            .expect("private-only file should index successfully");

        assert_eq!(stats.files_scanned, 1);
        assert_eq!(stats.files_indexed, 1);
        assert_eq!(stats.files_skipped, 0);
        assert_eq!(stats.symbols_extracted, 0);
        assert_eq!(stats.chunks_embedded, 0);
        assert_eq!(stats.languages.get(&Language::Rust), Some(&1));

        let state = storage.state.lock().expect("storage lock poisoned");
        assert!(state.stored_units.is_empty());
        assert_eq!(state.indexed_files.len(), 1);
        assert_eq!(
            state.indexed_files[0].file_path,
            PathBuf::from("src/internal.rs")
        );
        let metadata = state.metadata.clone().expect("metadata should be recorded");
        assert_eq!(metadata.file_count, 1);
        assert_eq!(metadata.symbol_count, 0);
    }

    #[tokio::test]
    async fn malformed_source_file_still_indexes_partial_results() {
        let temp_dir = TempDir::new().expect("temp dir should exist");
        fs::create_dir_all(temp_dir.path().join("src")).expect("src dir should exist");
        fs::write(
            temp_dir.path().join("src/broken.rs"),
            "/// Healthy function survives surrounding syntax noise.\n\
             pub fn healthy_example(value: &str) -> bool {\n\
                 value.contains(\"x\")\n\
             }\n\n\
             pub fn broken_example(value: &str) -> bool {\n\
                 value.contains(\"y\")\n",
        )
        .expect("broken source write should succeed");

        let storage = RecordingStorage::default();
        let stats = index_repo(temp_dir.path(), &storage, &FakeEmbedder)
            .await
            .expect("syntax errors should not abort indexing");

        assert_eq!(stats.files_scanned, 1);
        assert_eq!(stats.files_indexed, 1);
        assert_eq!(stats.files_skipped, 0);
        assert_eq!(stats.symbols_extracted, 1);
        assert_eq!(stats.chunks_embedded, 1);
        assert_eq!(stats.languages.get(&Language::Rust), Some(&1));

        let state = storage.state.lock().expect("storage lock poisoned");
        assert_eq!(state.stored_units.len(), 1);
        assert_eq!(state.stored_units[0].unit.name, "healthy_example");
        assert!(
            state.stored_units[0]
                .unit
                .file_path
                .ends_with(Path::new("src/broken.rs"))
        );
        let metadata = state.metadata.clone().expect("metadata should be recorded");
        assert_eq!(metadata.file_count, 1);
        assert_eq!(metadata.symbol_count, 1);
    }

    #[tokio::test]
    async fn indexer_preserves_unicode_source_paths() {
        let temp_dir = TempDir::new().expect("temp dir should exist");
        fs::create_dir_all(temp_dir.path().join("src")).expect("src dir should exist");
        let unicode_relative = PathBuf::from("src/naïve.rs");
        fs::write(
            temp_dir.path().join(&unicode_relative),
            "pub fn unicode_path() -> bool { true }\n",
        )
        .expect("unicode source write should succeed");

        let storage = RecordingStorage::default();
        let stats = index_repo(temp_dir.path(), &storage, &FakeEmbedder)
            .await
            .expect("unicode file paths should index successfully");

        assert_eq!(stats.files_scanned, 1);
        assert_eq!(stats.files_indexed, 1);
        let state = storage.state.lock().expect("storage lock poisoned");
        assert_eq!(state.stored_units.len(), 1);
        assert_eq!(state.stored_units[0].unit.name, "unicode_path");
        assert!(
            state.stored_units[0]
                .unit
                .file_path
                .ends_with(&unicode_relative)
        );
        assert!(
            state.indexed_files[0]
                .file_path
                .ends_with(&unicode_relative)
        );
    }

    fn git_repo_with_sources() -> TempDir {
        let temp_dir = TempDir::new().expect("temp dir should exist");
        init_empty_git_repo(temp_dir.path());
        temp_dir
    }

    async fn assert_fixture_index(fixture_name: &str, expected_files: u32, language: Language) {
        let fixture = TempGitFixture::new(fixture_name);
        let root = fixture.path_buf();
        let database = RecordingStorage::default();
        let context = detect_repo_context(&root).expect("fixture repo should resolve context");
        let stats = index_repo(&root, &database, &FakeEmbedder)
            .await
            .expect("fixture index should succeed");

        assert_eq!(stats.files_scanned, expected_files);
        assert_eq!(stats.files_indexed, expected_files);
        assert_eq!(stats.files_skipped, 0);
        assert_eq!(stats.languages.get(&language), Some(&expected_files));
        assert!(stats.duration_ms < 60_000);

        let expected_symbols_by_name = expected_symbol_names(fixture_name);
        let state = database.state.lock().expect("storage lock poisoned");
        let stored_units = state
            .stored_units
            .iter()
            .map(|entry| entry.unit.clone())
            .collect::<Vec<_>>();
        let indexed_files = state.indexed_files.clone();
        let metadata = state.metadata.clone().expect("metadata should be recorded");

        let actual_names = stored_units
            .iter()
            .map(|unit| unit.name.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(actual_names, expected_symbols_by_name);
        assert_eq!(stats.symbols_extracted, stored_units.len() as u32);
        assert_eq!(stats.chunks_embedded, stored_units.len() as u32);
        assert!(stored_units.iter().all(|unit| !unit.signature.is_empty()));
        assert_eq!(indexed_files.len(), expected_files as usize);
        assert_eq!(
            state
                .stored_units
                .iter()
                .filter(|entry| entry.embedding.is_some())
                .count() as u32,
            stored_units.len() as u32
        );

        assert_eq!(metadata.file_count, expected_files);
        assert_eq!(metadata.symbol_count, stored_units.len() as u32);
        assert_eq!(metadata.repo_id, context.repo_id);
        assert!(!metadata.branch.is_empty());
        assert!(metadata.last_commit_sha.is_some());
        assert!(metadata.last_indexed_at <= Utc::now());
    }

    fn expected_symbol_names(fixture_name: &str) -> BTreeSet<String> {
        let path = git_fixture::fixture_path(fixture_name).join("expected.json");
        let value: serde_json::Value =
            serde_json::from_slice(&fs::read(path).expect("expected fixture should exist"))
                .expect("expected fixture should parse");

        value["symbols"]
            .as_array()
            .expect("symbols should be an array")
            .iter()
            .map(|symbol| {
                symbol["name"]
                    .as_str()
                    .expect("symbol name should be a string")
                    .to_string()
            })
            .collect()
    }
}
