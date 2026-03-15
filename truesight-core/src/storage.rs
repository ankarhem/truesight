use async_trait::async_trait;
use std::path::{Path, PathBuf};

use crate::{
    CodeUnit, EmbeddingUpdate, IndexMetadata, IndexedCodeUnit, IndexedFileRecord, PendingEmbedding,
    RankedResult, Result,
};

#[cfg_attr(feature = "mocking", mockall::automock)]
#[async_trait]
pub trait Storage: Send + Sync {
    async fn store_code_units(&self, repo_id: &str, branch: &str, units: &[CodeUnit])
    -> Result<()>;

    async fn search_fts(
        &self,
        repo_id: &str,
        branch: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RankedResult>>;

    async fn search_vector(
        &self,
        repo_id: &str,
        branch: &str,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<RankedResult>>;

    async fn search_hybrid(
        &self,
        repo_id: &str,
        branch: &str,
        query: &str,
        embedding: &[f32],
        limit: usize,
        rrf_k: u32,
    ) -> Result<Vec<RankedResult>>;

    async fn get_index_metadata(
        &self,
        repo_id: &str,
        branch: &str,
    ) -> Result<Option<IndexMetadata>>;

    async fn has_indexed_symbols(&self, repo_id: &str, branch: &str) -> Result<bool>;

    async fn set_index_metadata(
        &self,
        repo_id: &str,
        branch: &str,
        meta: &IndexMetadata,
    ) -> Result<()>;

    async fn delete_branch_index(&self, repo_id: &str, branch: &str) -> Result<()>;

    async fn get_all_symbols(&self, repo_id: &str, branch: &str) -> Result<Vec<CodeUnit>>;
}

#[async_trait]
pub trait IndexStorage: Storage {
    async fn store_indexed_units(
        &self,
        repo_id: &str,
        branch: &str,
        units: &[IndexedCodeUnit],
    ) -> Result<()>;

    async fn upsert_indexed_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
        file_hash: &str,
        chunk_count: u32,
    ) -> Result<()>;

    async fn list_pending_embeddings(
        &self,
        repo_id: &str,
        branch: &str,
        embedding_model: &str,
        limit: usize,
    ) -> Result<Vec<PendingEmbedding>>;

    async fn update_embeddings(
        &self,
        repo_id: &str,
        branch: &str,
        embedding_model: &str,
        updates: &[EmbeddingUpdate],
    ) -> Result<()>;

    async fn replace_branch_index(
        &self,
        repo_id: &str,
        branch: &str,
        units: &[IndexedCodeUnit],
        indexed_files: &[IndexedFileRecord],
        metadata: &IndexMetadata,
    ) -> Result<()> {
        self.delete_branch_index(repo_id, branch).await?;

        if !units.is_empty() {
            self.store_indexed_units(repo_id, branch, units).await?;
        }

        for file in indexed_files {
            self.upsert_indexed_file(
                repo_id,
                branch,
                &file.file_path,
                &file.file_hash,
                file.chunk_count,
            )
            .await?;
        }

        self.set_index_metadata(repo_id, branch, metadata).await
    }
}

#[cfg(feature = "mocking")]
mockall::mock! {
    pub IndexStorage {}

    #[async_trait]
    impl Storage for IndexStorage {
        async fn store_code_units(&self, repo_id: &str, branch: &str, units: &[CodeUnit])
            -> Result<()>;
        async fn search_fts(
            &self,
            repo_id: &str,
            branch: &str,
            query: &str,
            limit: usize,
        ) -> Result<Vec<RankedResult>>;
        async fn search_vector(
            &self,
            repo_id: &str,
            branch: &str,
            embedding: &[f32],
            limit: usize,
        ) -> Result<Vec<RankedResult>>;
        async fn search_hybrid(
            &self,
            repo_id: &str,
            branch: &str,
            query: &str,
            embedding: &[f32],
            limit: usize,
            rrf_k: u32,
        ) -> Result<Vec<RankedResult>>;
        async fn get_index_metadata(
            &self,
            repo_id: &str,
            branch: &str,
        ) -> Result<Option<IndexMetadata>>;
        async fn has_indexed_symbols(&self, repo_id: &str, branch: &str) -> Result<bool>;
        async fn set_index_metadata(
            &self,
            repo_id: &str,
            branch: &str,
            meta: &IndexMetadata,
        ) -> Result<()>;
        async fn delete_branch_index(&self, repo_id: &str, branch: &str) -> Result<()>;
        async fn get_all_symbols(&self, repo_id: &str, branch: &str) -> Result<Vec<CodeUnit>>;
    }

    #[async_trait]
    impl IndexStorage for IndexStorage {
        async fn store_indexed_units(
            &self,
            repo_id: &str,
            branch: &str,
            units: &[IndexedCodeUnit],
        ) -> Result<()>;
        async fn upsert_indexed_file(
            &self,
            repo_id: &str,
            branch: &str,
            file_path: &Path,
            file_hash: &str,
            chunk_count: u32,
        ) -> Result<()>;
        async fn list_pending_embeddings(
            &self,
            repo_id: &str,
            branch: &str,
            embedding_model: &str,
            limit: usize,
        ) -> Result<Vec<PendingEmbedding>>;
        async fn update_embeddings(
            &self,
            repo_id: &str,
            branch: &str,
            embedding_model: &str,
            updates: &[EmbeddingUpdate],
        ) -> Result<()>;
    }
}

#[async_trait]
pub trait IncrementalStorage: IndexStorage {
    async fn get_indexed_files(
        &self,
        repo_id: &str,
        branch: &str,
    ) -> Result<Vec<IndexedFileRecord>>;

    async fn delete_units_for_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
    ) -> Result<()>;

    async fn delete_indexed_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
    ) -> Result<()>;

    async fn apply_incremental_changes(
        &self,
        repo_id: &str,
        branch: &str,
        deleted_files: &[PathBuf],
        units: &[IndexedCodeUnit],
        indexed_files: &[IndexedFileRecord],
    ) -> Result<()> {
        for file_path in deleted_files {
            self.delete_units_for_file(repo_id, branch, file_path)
                .await?;
            self.delete_indexed_file(repo_id, branch, file_path).await?;
        }

        if !units.is_empty() {
            self.store_indexed_units(repo_id, branch, units).await?;
        }

        for file in indexed_files {
            self.upsert_indexed_file(
                repo_id,
                branch,
                &file.file_path,
                &file.file_hash,
                file.chunk_count,
            )
            .await?;
        }

        Ok(())
    }
}

#[cfg(feature = "mocking")]
mockall::mock! {
    pub IncrementalStorage {}

    #[async_trait]
    impl Storage for IncrementalStorage {
        async fn store_code_units(&self, repo_id: &str, branch: &str, units: &[CodeUnit])
            -> Result<()>;
        async fn search_fts(
            &self,
            repo_id: &str,
            branch: &str,
            query: &str,
            limit: usize,
        ) -> Result<Vec<RankedResult>>;
        async fn search_vector(
            &self,
            repo_id: &str,
            branch: &str,
            embedding: &[f32],
            limit: usize,
        ) -> Result<Vec<RankedResult>>;
        async fn search_hybrid(
            &self,
            repo_id: &str,
            branch: &str,
            query: &str,
            embedding: &[f32],
            limit: usize,
            rrf_k: u32,
        ) -> Result<Vec<RankedResult>>;
        async fn get_index_metadata(
            &self,
            repo_id: &str,
            branch: &str,
        ) -> Result<Option<IndexMetadata>>;
        async fn has_indexed_symbols(&self, repo_id: &str, branch: &str) -> Result<bool>;
        async fn set_index_metadata(
            &self,
            repo_id: &str,
            branch: &str,
            meta: &IndexMetadata,
        ) -> Result<()>;
        async fn delete_branch_index(&self, repo_id: &str, branch: &str) -> Result<()>;
        async fn get_all_symbols(&self, repo_id: &str, branch: &str) -> Result<Vec<CodeUnit>>;
    }

    #[async_trait]
    impl IndexStorage for IncrementalStorage {
        async fn store_indexed_units(
            &self,
            repo_id: &str,
            branch: &str,
            units: &[IndexedCodeUnit],
        ) -> Result<()>;
        async fn upsert_indexed_file(
            &self,
            repo_id: &str,
            branch: &str,
            file_path: &Path,
            file_hash: &str,
            chunk_count: u32,
        ) -> Result<()>;
        async fn list_pending_embeddings(
            &self,
            repo_id: &str,
            branch: &str,
            embedding_model: &str,
            limit: usize,
        ) -> Result<Vec<PendingEmbedding>>;
        async fn update_embeddings(
            &self,
            repo_id: &str,
            branch: &str,
            embedding_model: &str,
            updates: &[EmbeddingUpdate],
        ) -> Result<()>;
    }

    #[async_trait]
    impl IncrementalStorage for IncrementalStorage {
        async fn get_indexed_files(
            &self,
            repo_id: &str,
            branch: &str,
        ) -> Result<Vec<IndexedFileRecord>>;
        async fn delete_units_for_file(
            &self,
            repo_id: &str,
            branch: &str,
            file_path: &Path,
        ) -> Result<()>;
        async fn delete_indexed_file(
            &self,
            repo_id: &str,
            branch: &str,
            file_path: &Path,
        ) -> Result<()>;
    }
}

#[cfg_attr(feature = "mocking", mockall::automock)]
pub trait Embedder: Send + Sync {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    fn embed_batch<'a>(&self, texts: &[&'a str]) -> Result<Vec<Vec<f32>>>;

    fn dimension(&self) -> usize;

    fn model_name(&self) -> &str {
        "unknown"
    }
}
