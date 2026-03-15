use std::path::Path;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::Utc;
use truesight_core::{
    CodeUnit, Embedder, EmbeddingUpdate, IncrementalStorage, IndexMetadata, IndexStorage,
    IndexedCodeUnit, IndexedFileRecord, PendingEmbedding, RankedResult, Storage,
};

#[derive(Clone, Default)]
pub struct RecordingStorage {
    pub state: Arc<Mutex<StorageState>>,
}

#[derive(Default)]
pub struct StorageState {
    pub stored_units: Vec<IndexedCodeUnit>,
    pub indexed_files: Vec<IndexedFileRecord>,
    pub metadata: Option<IndexMetadata>,
}

#[async_trait]
impl Storage for RecordingStorage {
    async fn store_code_units(
        &self,
        _repo_id: &str,
        _branch: &str,
        units: &[CodeUnit],
    ) -> truesight_core::Result<()> {
        self.state
            .lock()
            .expect("storage lock poisoned")
            .stored_units
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
            .stored_units
            .is_empty())
    }

    async fn delete_branch_index(
        &self,
        _repo_id: &str,
        _branch: &str,
    ) -> truesight_core::Result<()> {
        let mut state = self.state.lock().expect("storage lock poisoned");
        state.stored_units.clear();
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
            .stored_units
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
            .stored_units
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
            .stored_units
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
                .stored_units
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
            .stored_units
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

pub struct DeterministicFakeEmbedder;

impl Embedder for DeterministicFakeEmbedder {
    fn embed(&self, text: &str) -> truesight_core::Result<Vec<f32>> {
        Ok(vec![
            text.len() as f32,
            text.bytes().map(u32::from).sum::<u32>() as f32,
            text.lines().count() as f32,
            if text.contains("pub") { 1.0 } else { 0.0 },
        ])
    }

    fn embed_batch<'a>(&self, texts: &[&'a str]) -> truesight_core::Result<Vec<Vec<f32>>> {
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
