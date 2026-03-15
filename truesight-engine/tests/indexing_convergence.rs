use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use tempfile::TempDir;
use truesight_core::{
    CodeUnit, Embedder, EmbeddingUpdate, IncrementalStorage, IndexMetadata, IndexStats,
    IndexStorage, IndexedCodeUnit, IndexedFileRecord, PendingEmbedding, RankedResult, Storage,
};
use truesight_engine::{IncrementalIndexer, detect_repo_context, index_repo};

#[path = "test_support/git_fixture.rs"]
mod git_fixture;

use git_fixture::{TempGitFixture, init_git_repo};

#[derive(Clone, Default)]
struct RecordingStorage {
    state: Arc<Mutex<StorageState>>,
}

#[derive(Default)]
struct StorageState {
    units: Vec<IndexedCodeUnit>,
    indexed_files: Vec<IndexedFileRecord>,
    metadata: Option<IndexMetadata>,
}

#[async_trait]
impl Storage for RecordingStorage {
    async fn store_code_units(
        &self,
        _repo_id: &str,
        _branch: &str,
        _units: &[CodeUnit],
    ) -> truesight_core::Result<()> {
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

    async fn search_hybrid(
        &self,
        _repo_id: &str,
        _branch: &str,
        _query: &str,
        _embedding: &[f32],
        _limit: usize,
        _rrf_k: u32,
    ) -> truesight_core::Result<Vec<RankedResult>> {
        Ok(Vec::new())
    }

    async fn get_index_metadata(
        &self,
        _repo_id: &str,
        _branch: &str,
    ) -> truesight_core::Result<Option<IndexMetadata>> {
        Ok(self
            .state
            .lock()
            .expect("storage lock poisoned")
            .metadata
            .clone())
    }

    async fn set_index_metadata(
        &self,
        _repo_id: &str,
        _branch: &str,
        meta: &IndexMetadata,
    ) -> truesight_core::Result<()> {
        self.state.lock().expect("storage lock poisoned").metadata = Some(meta.clone());
        Ok(())
    }

    async fn has_indexed_symbols(
        &self,
        _repo_id: &str,
        _branch: &str,
    ) -> truesight_core::Result<bool> {
        Ok(!self
            .state
            .lock()
            .expect("storage lock poisoned")
            .units
            .is_empty())
    }

    async fn delete_branch_index(
        &self,
        _repo_id: &str,
        _branch: &str,
    ) -> truesight_core::Result<()> {
        let mut state = self.state.lock().expect("storage lock poisoned");
        state.units.clear();
        state.indexed_files.clear();
        state.metadata = None;
        Ok(())
    }

    async fn get_all_symbols(
        &self,
        _repo_id: &str,
        _branch: &str,
    ) -> truesight_core::Result<Vec<CodeUnit>> {
        Ok(self
            .state
            .lock()
            .expect("storage lock poisoned")
            .units
            .iter()
            .map(|entry| entry.unit.clone())
            .collect())
    }
}

#[async_trait]
impl IndexStorage for RecordingStorage {
    async fn store_indexed_units(
        &self,
        _repo_id: &str,
        _branch: &str,
        units: &[IndexedCodeUnit],
    ) -> truesight_core::Result<()> {
        self.state
            .lock()
            .expect("storage lock poisoned")
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
        chunk_count: u32,
    ) -> truesight_core::Result<()> {
        let mut state = self.state.lock().expect("storage lock poisoned");
        state
            .indexed_files
            .retain(|record| record.file_path != file_path);
        state.indexed_files.push(IndexedFileRecord {
            file_path: file_path.to_path_buf(),
            file_hash: file_hash.to_string(),
            chunk_count,
            indexed_at: Utc::now(),
        });
        state
            .indexed_files
            .sort_by(|left, right| left.file_path.cmp(&right.file_path));
        Ok(())
    }

    async fn list_pending_embeddings(
        &self,
        _repo_id: &str,
        _branch: &str,
        _embedding_model: &str,
        limit: usize,
    ) -> truesight_core::Result<Vec<PendingEmbedding>> {
        Ok(self
            .state
            .lock()
            .expect("storage lock poisoned")
            .units
            .iter()
            .filter(|entry| entry.embedding.is_none())
            .take(limit)
            .map(|entry| PendingEmbedding {
                id: pending_id(&entry.unit),
                signature: entry.unit.signature.clone(),
                doc: entry.unit.doc.clone(),
                content: entry.unit.content.clone(),
            })
            .collect())
    }

    async fn update_embeddings(
        &self,
        _repo_id: &str,
        _branch: &str,
        _embedding_model: &str,
        updates: &[EmbeddingUpdate],
    ) -> truesight_core::Result<()> {
        let mut state = self.state.lock().expect("storage lock poisoned");
        for update in updates {
            if let Some(entry) = state
                .units
                .iter_mut()
                .find(|entry| pending_id(&entry.unit) == update.id)
            {
                entry.embedding = Some(update.embedding.clone());
            }
        }
        Ok(())
    }
}

#[async_trait]
impl IncrementalStorage for RecordingStorage {
    async fn get_indexed_files(
        &self,
        _repo_id: &str,
        _branch: &str,
    ) -> truesight_core::Result<Vec<IndexedFileRecord>> {
        Ok(self
            .state
            .lock()
            .expect("storage lock poisoned")
            .indexed_files
            .clone())
    }

    async fn delete_units_for_file(
        &self,
        _repo_id: &str,
        _branch: &str,
        file_path: &Path,
    ) -> truesight_core::Result<()> {
        self.state
            .lock()
            .expect("storage lock poisoned")
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
            .expect("storage lock poisoned")
            .indexed_files
            .retain(|record| record.file_path != file_path);
        Ok(())
    }
}

struct FakeEmbedder;

impl Embedder for FakeEmbedder {
    fn embed(&self, text: &str) -> truesight_core::Result<Vec<f32>> {
        Ok(vec![
            text.len() as f32,
            text.bytes().map(u32::from).sum::<u32>() as f32,
            text.lines().count() as f32,
            if text.contains("pub") { 1.0 } else { 0.0 },
        ])
    }

    fn embed_batch(&self, texts: &[&str]) -> truesight_core::Result<Vec<Vec<f32>>> {
        texts.iter().map(|text| self.embed(text)).collect()
    }

    fn dimension(&self) -> usize {
        4
    }

    fn model_name(&self) -> &str {
        "fake-model"
    }
}

fn pending_id(unit: &CodeUnit) -> String {
    format!(
        "{}:{}:{}",
        unit.file_path.display(),
        unit.name,
        unit.line_start
    )
}

#[derive(Debug, Clone, PartialEq)]
struct ComparableIndexedUnit {
    file_path: PathBuf,
    name: String,
    kind: String,
    signature: String,
    doc: Option<String>,
    line_start: u32,
    line_end: u32,
    content: String,
    parent: Option<String>,
    language: String,
    embedding: Option<Vec<f32>>,
    file_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ComparableIndexedFile {
    file_path: PathBuf,
    file_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ComparableMetadata {
    repo_id: String,
    branch: String,
    last_commit_sha: Option<String>,
    file_count: u32,
    symbol_count: u32,
}

#[tokio::test]
async fn full_and_incremental_indexing_store_equivalent_results_for_same_repo_state() {
    let fixture = TempGitFixture::new("rust-fixture");
    let (full_stats, full_storage, incremental_stats, incremental_storage) =
        index_with_full_and_incremental(fixture.path()).await;

    assert_eq!(
        comparable_stats(&full_stats),
        comparable_stats(&incremental_stats)
    );
    assert_eq!(
        stored_units(&full_storage),
        stored_units(&incremental_storage)
    );
    assert_eq!(
        indexed_files(&full_storage),
        indexed_files(&incremental_storage)
    );
    assert_eq!(metadata(&full_storage), metadata(&incremental_storage));
}

#[tokio::test]
async fn full_and_incremental_indexing_agree_on_private_only_supported_files() {
    let temp = TempDir::new().expect("temp dir should exist");
    let repo_root = temp.path().join("repo");
    write_file(
        &repo_root,
        "src/internal.rs",
        "fn hidden_example() -> bool { true }\n",
    );
    init_git_repo(&repo_root, "initial private file");

    let (full_stats, full_storage, incremental_stats, incremental_storage) =
        index_with_full_and_incremental(&repo_root).await;

    assert_eq!(
        comparable_stats(&full_stats),
        comparable_stats(&incremental_stats)
    );
    assert!(stored_units(&full_storage).is_empty());
    assert!(stored_units(&incremental_storage).is_empty());
    assert_eq!(indexed_files(&full_storage).len(), 1);
    assert_eq!(indexed_files(&incremental_storage).len(), 1);
    assert_eq!(metadata(&full_storage).file_count, 1);
    assert_eq!(metadata(&incremental_storage).file_count, 1);
    assert_eq!(metadata(&full_storage).symbol_count, 0);
    assert_eq!(metadata(&incremental_storage).symbol_count, 0);
}

async fn index_with_full_and_incremental(
    root: &Path,
) -> (IndexStats, RecordingStorage, IndexStats, RecordingStorage) {
    let context = detect_repo_context(root).expect("repo context should resolve");
    let full_storage = RecordingStorage::default();
    let incremental_storage = RecordingStorage::default();
    let embedder = FakeEmbedder;

    let full_stats = index_repo(root, &full_storage, &embedder)
        .await
        .expect("full indexing should succeed");

    let indexer = IncrementalIndexer::new();
    let changes = indexer
        .detect_changes(
            root,
            &incremental_storage,
            &context.repo_id,
            &context.branch,
        )
        .await
        .expect("initial change detection should succeed");
    let incremental_stats = indexer
        .incremental_index(
            root,
            &changes,
            &incremental_storage,
            &embedder,
            &context.repo_id,
            &context.branch,
        )
        .await
        .expect("incremental indexing should succeed");

    (
        full_stats,
        full_storage,
        incremental_stats,
        incremental_storage,
    )
}

fn comparable_stats(stats: &IndexStats) -> (u32, u32, u32, u32, u32, Vec<(String, u32)>) {
    let mut languages = stats
        .languages
        .iter()
        .map(|(language, count)| (language.to_string(), *count))
        .collect::<Vec<_>>();
    languages.sort();

    (
        stats.files_scanned,
        stats.files_indexed,
        stats.files_skipped,
        stats.symbols_extracted,
        stats.chunks_embedded,
        languages,
    )
}

fn stored_units(storage: &RecordingStorage) -> Vec<ComparableIndexedUnit> {
    let mut units = storage
        .state
        .lock()
        .expect("storage lock poisoned")
        .units
        .iter()
        .map(|entry| ComparableIndexedUnit {
            file_path: entry.unit.file_path.clone(),
            name: entry.unit.name.clone(),
            kind: entry.unit.kind.to_string(),
            signature: entry.unit.signature.clone(),
            doc: entry.unit.doc.clone(),
            line_start: entry.unit.line_start,
            line_end: entry.unit.line_end,
            content: entry.unit.content.clone(),
            parent: entry.unit.parent.clone(),
            language: entry.unit.language.to_string(),
            embedding: entry.embedding.clone(),
            file_hash: entry.file_hash.clone(),
        })
        .collect::<Vec<_>>();
    units.sort_by(|left, right| {
        left.file_path
            .cmp(&right.file_path)
            .then(left.line_start.cmp(&right.line_start))
            .then(left.name.cmp(&right.name))
    });
    units
}

fn indexed_files(storage: &RecordingStorage) -> Vec<ComparableIndexedFile> {
    let mut files = storage
        .state
        .lock()
        .expect("storage lock poisoned")
        .indexed_files
        .iter()
        .map(|record| ComparableIndexedFile {
            file_path: record.file_path.clone(),
            file_hash: record.file_hash.clone(),
        })
        .collect::<Vec<_>>();
    files.sort_by(|left, right| left.file_path.cmp(&right.file_path));
    files
}

fn metadata(storage: &RecordingStorage) -> ComparableMetadata {
    let metadata = storage
        .state
        .lock()
        .expect("storage lock poisoned")
        .metadata
        .clone()
        .expect("metadata should be present");
    ComparableMetadata {
        repo_id: metadata.repo_id,
        branch: metadata.branch,
        last_commit_sha: metadata.last_commit_sha,
        file_count: metadata.file_count,
        symbol_count: metadata.symbol_count,
    }
}

fn write_file(root: &Path, relative: &str, contents: &str) {
    let path = root.join(relative);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("parent dir should exist");
    }
    std::fs::write(path, contents).expect("fixture file should write");
}
