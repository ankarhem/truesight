use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use libsql::{Connection, Database as LibsqlDatabase, params};
use sha2::{Digest, Sha256};
use tracing::warn;
use truesight_core::{
    CodeUnit, EmbeddingUpdate, IncrementalStorage, IndexMetadata, IndexStatus, IndexStorage,
    IndexedCodeUnit, IndexedFileRecord, MatchType, PendingEmbedding, RankedResult, Result, Storage,
    TruesightError,
};

mod error;
mod helpers;
mod rows;
mod schema;
mod search;

use error::DatabaseError;
use helpers::{
    code_unit_id, cosine_similarity, decode_embedding, default_file_hash, encode_embedding,
    hybrid_candidate_limit, normalize_fts_score, normalize_repo_root, parse_timestamp,
    path_to_string, sanitize_fts_query, sanitize_path_segment, short_hex_digest,
};
use rows::{code_unit_from_row, insert_code_unit};
use schema::{MIGRATION_TABLE_SQL, ensure_vector_index, run_incremental_migrations};
use search::{fuse_ranked_results, ranked_result_from_row};

#[derive(Clone)]
pub struct Database {
    db: Arc<LibsqlDatabase>,
}

impl Database {
    pub async fn new(db_path: &Path) -> std::result::Result<Self, DatabaseError> {
        if db_path != Path::new(":memory:") {
            if let Some(parent) = db_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
        }

        let db = libsql::Builder::new_local(db_path).build().await?;
        Ok(Self { db: Arc::new(db) })
    }

    pub async fn connect(&self) -> std::result::Result<Connection, DatabaseError> {
        let connection = self.db.connect()?;
        connection.query("PRAGMA busy_timeout = 5000", ()).await?;
        Ok(connection)
    }

    pub async fn run_migrations(&self) -> std::result::Result<(), DatabaseError> {
        let connection = self.connect().await?;
        connection.query("PRAGMA journal_mode=WAL", ()).await?;
        connection.execute_batch(MIGRATION_TABLE_SQL).await?;
        run_incremental_migrations(&connection).await?;
        ensure_vector_index(&connection).await
    }

    pub fn config_dir() -> std::result::Result<PathBuf, DatabaseError> {
        let home = std::env::var_os("HOME").ok_or(DatabaseError::MissingHomeDirectory)?;
        let config_dir = PathBuf::from(home).join(".config").join("truesight");
        std::fs::create_dir_all(&config_dir)?;
        Ok(config_dir)
    }

    pub fn db_path_for_repo(repo_root: &Path) -> std::result::Result<PathBuf, DatabaseError> {
        let normalized_repo_root = normalize_repo_root(repo_root)?;
        let repo_name = normalized_repo_root
            .file_name()
            .and_then(|segment| segment.to_str())
            .filter(|segment| !segment.is_empty())
            .map(sanitize_path_segment)
            .unwrap_or_else(|| String::from("repo"));
        let digest = short_hex_digest(normalized_repo_root.to_string_lossy().as_bytes());

        Ok(Self::config_dir()?.join(format!("{repo_name}-{digest}.db")))
    }

    pub async fn store_indexed_units(
        &self,
        repo_id: &str,
        branch: &str,
        units: &[IndexedCodeUnit],
    ) -> Result<()> {
        if units.is_empty() {
            return Ok(());
        }

        let connection = self.connect().await.map_err(TruesightError::from)?;
        let transaction = connection
            .transaction()
            .await
            .map_err(DatabaseError::from)?;

        for entry in units {
            insert_code_unit(&transaction, repo_id, branch, entry)
                .await
                .map_err(TruesightError::from)?;
        }

        transaction.commit().await.map_err(DatabaseError::from)?;
        Ok(())
    }

    pub async fn upsert_indexed_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
        file_hash: &str,
        chunk_count: u32,
    ) -> Result<()> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        connection
            .execute(
                r#"
                INSERT INTO indexed_files (repo_id, branch, file_path, file_hash, chunk_count)
                VALUES (?1, ?2, ?3, ?4, ?5)
                ON CONFLICT(repo_id, branch, file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    chunk_count = excluded.chunk_count,
                    indexed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                "#,
                params![
                    repo_id,
                    branch,
                    path_to_string(file_path),
                    file_hash,
                    i64::from(chunk_count),
                ],
            )
            .await
            .map_err(DatabaseError::from)?;
        Ok(())
    }

    pub async fn update_index_status(
        &self,
        repo_id: &str,
        branch: &str,
        status: IndexStatus,
        last_seen_commit_sha: Option<&str>,
        embedding_model: Option<&str>,
        last_error: Option<&str>,
    ) -> Result<()> {
        let mut metadata = self
            .get_index_metadata(repo_id, branch)
            .await?
            .unwrap_or_else(|| IndexMetadata {
                repo_id: repo_id.to_string(),
                branch: branch.to_string(),
                status: IndexStatus::Ready,
                last_indexed_at: Utc::now(),
                last_commit_sha: None,
                last_seen_commit_sha: None,
                file_count: 0,
                symbol_count: 0,
                embedding_model: String::new(),
                last_error: None,
            });

        metadata.status = status;
        metadata.last_seen_commit_sha = last_seen_commit_sha.map(ToOwned::to_owned);
        metadata.last_error = last_error.map(ToOwned::to_owned);

        if let Some(model) = embedding_model {
            metadata.embedding_model = model.to_string();
        }

        if status == IndexStatus::Ready {
            metadata.last_indexed_at = Utc::now();
            metadata.last_error = None;
        }

        self.set_index_metadata(repo_id, branch, &metadata).await
    }

    pub async fn replace_branch_index(
        &self,
        repo_id: &str,
        branch: &str,
        units: &[IndexedCodeUnit],
        indexed_files: &[IndexedFileRecord],
        metadata: &IndexMetadata,
    ) -> Result<()> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let transaction = connection
            .transaction()
            .await
            .map_err(DatabaseError::from)?;

        transaction
            .execute(
                "DELETE FROM code_units WHERE repo_id = ?1 AND branch = ?2",
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;
        transaction
            .execute(
                "DELETE FROM indexed_files WHERE repo_id = ?1 AND branch = ?2",
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;
        transaction
            .execute(
                "DELETE FROM index_metadata WHERE repo_id = ?1 AND branch = ?2",
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;

        for entry in units {
            insert_code_unit(&transaction, repo_id, branch, entry)
                .await
                .map_err(TruesightError::from)?;
        }

        for file in indexed_files {
            transaction
                .execute(
                    r#"
                    INSERT INTO indexed_files (repo_id, branch, file_path, file_hash, chunk_count)
                    VALUES (?1, ?2, ?3, ?4, ?5)
                    ON CONFLICT(repo_id, branch, file_path) DO UPDATE SET
                        file_hash = excluded.file_hash,
                        chunk_count = excluded.chunk_count,
                        indexed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    "#,
                    params![
                        repo_id,
                        branch,
                        path_to_string(&file.file_path),
                        file.file_hash.clone(),
                        i64::from(file.chunk_count),
                    ],
                )
                .await
                .map_err(DatabaseError::from)?;
        }

        transaction
            .execute(
                r#"
                INSERT INTO index_metadata (
                    repo_id,
                    branch,
                    status,
                    last_indexed_at,
                    last_commit_sha,
                    last_seen_commit_sha,
                    file_count,
                    symbol_count,
                    embedding_model,
                    last_error
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                ON CONFLICT(repo_id, branch) DO UPDATE SET
                    status = excluded.status,
                    last_indexed_at = excluded.last_indexed_at,
                    last_commit_sha = excluded.last_commit_sha,
                    last_seen_commit_sha = excluded.last_seen_commit_sha,
                    file_count = excluded.file_count,
                    symbol_count = excluded.symbol_count,
                    embedding_model = excluded.embedding_model,
                    last_error = excluded.last_error
                "#,
                params![
                    repo_id,
                    branch,
                    metadata.status.to_string(),
                    metadata.last_indexed_at.to_rfc3339(),
                    metadata.last_commit_sha.clone(),
                    metadata.last_seen_commit_sha.clone(),
                    i64::from(metadata.file_count),
                    i64::from(metadata.symbol_count),
                    metadata.embedding_model.clone(),
                    metadata.last_error.clone(),
                ],
            )
            .await
            .map_err(DatabaseError::from)?;

        transaction.commit().await.map_err(DatabaseError::from)?;
        Ok(())
    }

    pub async fn get_indexed_files(
        &self,
        repo_id: &str,
        branch: &str,
    ) -> Result<Vec<IndexedFileRecord>> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let mut rows = connection
            .query(
                r#"
                SELECT file_path, file_hash, chunk_count, indexed_at
                FROM indexed_files
                WHERE repo_id = ?1 AND branch = ?2
                ORDER BY file_path
                "#,
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;
        let mut files = Vec::new();

        while let Some(row) = rows.next().await.map_err(DatabaseError::from)? {
            files.push(IndexedFileRecord {
                file_path: PathBuf::from(row.get::<String>(0).map_err(DatabaseError::from)?),
                file_hash: row.get::<String>(1).map_err(DatabaseError::from)?,
                chunk_count: row.get::<i64>(2).map_err(DatabaseError::from)? as u32,
                indexed_at: parse_timestamp(&row.get::<String>(3).map_err(DatabaseError::from)?)?,
            });
        }

        Ok(files)
    }

    pub async fn has_indexed_symbols(&self, repo_id: &str, branch: &str) -> Result<bool> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let mut rows = connection
            .query(
                r#"
                SELECT 1
                FROM code_units
                WHERE repo_id = ?1 AND branch = ?2
                LIMIT 1
                "#,
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;
        Ok(rows.next().await.map_err(DatabaseError::from)?.is_some())
    }

    pub async fn list_pending_embeddings(
        &self,
        repo_id: &str,
        branch: &str,
        embedding_model: &str,
        limit: usize,
    ) -> Result<Vec<PendingEmbedding>> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let mut rows = connection
            .query(
                r#"
                SELECT id, signature, doc, content
                FROM code_units
                WHERE repo_id = ?1
                  AND branch = ?2
                  AND (embedding IS NULL OR embedding_model != ?3)
                ORDER BY file_path ASC, line_start ASC, name ASC
                LIMIT ?4
                "#,
                params![repo_id, branch, embedding_model, limit as i64],
            )
            .await
            .map_err(DatabaseError::from)?;
        let mut pending = Vec::new();

        while let Some(row) = rows.next().await.map_err(DatabaseError::from)? {
            pending.push(PendingEmbedding {
                id: row.get::<String>(0).map_err(DatabaseError::from)?,
                signature: row.get::<String>(1).map_err(DatabaseError::from)?,
                doc: row.get::<Option<String>>(2).map_err(DatabaseError::from)?,
                content: row.get::<String>(3).map_err(DatabaseError::from)?,
            });
        }

        Ok(pending)
    }

    pub async fn update_embeddings(
        &self,
        repo_id: &str,
        branch: &str,
        embedding_model: &str,
        updates: &[EmbeddingUpdate],
    ) -> Result<()> {
        if updates.is_empty() {
            return Ok(());
        }

        let connection = self.connect().await.map_err(TruesightError::from)?;
        let transaction = connection
            .transaction()
            .await
            .map_err(DatabaseError::from)?;

        for update in updates {
            transaction
                .execute(
                    r#"
                    UPDATE code_units
                    SET embedding = ?1, embedding_model = ?2
                    WHERE repo_id = ?3 AND branch = ?4 AND id = ?5
                    "#,
                    params![
                        encode_embedding(&update.embedding)?,
                        embedding_model,
                        repo_id,
                        branch,
                        update.id.clone(),
                    ],
                )
                .await
                .map_err(DatabaseError::from)?;
        }

        transaction.commit().await.map_err(DatabaseError::from)?;
        Ok(())
    }

    #[cfg(test)]
    async fn list_branches(&self, repo_id: &str) -> Result<Vec<String>> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let mut rows = connection
            .query(
                r#"
                SELECT DISTINCT branch
                FROM index_metadata
                WHERE repo_id = ?1
                ORDER BY branch ASC
                "#,
                params![repo_id],
            )
            .await
            .map_err(DatabaseError::from)?;
        let mut branches = Vec::new();

        while let Some(row) = rows.next().await.map_err(DatabaseError::from)? {
            branches.push(row.get::<String>(0).map_err(DatabaseError::from)?);
        }

        Ok(branches)
    }

    #[cfg(test)]
    async fn cleanup_branch(&self, repo_id: &str, branch: &str) -> Result<()> {
        self.delete_branch_index(repo_id, branch).await
    }

    pub async fn delete_units_for_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
    ) -> Result<()> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let transaction = connection
            .transaction()
            .await
            .map_err(DatabaseError::from)?;
        transaction
            .execute(
                "DELETE FROM code_units WHERE repo_id = ?1 AND branch = ?2 AND file_path = ?3",
                params![repo_id, branch, path_to_string(file_path)],
            )
            .await
            .map_err(DatabaseError::from)?;
        transaction.commit().await.map_err(DatabaseError::from)?;
        Ok(())
    }

    async fn try_vector_top_k(
        &self,
        repo_id: &str,
        branch: &str,
        embedding: &[f32],
        limit: usize,
    ) -> std::result::Result<Vec<RankedResult>, DatabaseError> {
        let connection = self.connect().await?;
        let query_blob = encode_embedding(embedding)?;
        let mut rows = connection
            .query(
                r#"
                SELECT
                    code_units.name,
                    code_units.kind,
                    code_units.signature,
                    code_units.doc,
                    code_units.file_path,
                    code_units.line_start,
                    code_units.line_end,
                    code_units.content,
                    code_units.parent,
                    code_units.language,
                    v.distance
                FROM vector_top_k('idx_code_units_embedding', vector(?1), ?2) AS v
                JOIN code_units ON code_units._rowid = v.id
                WHERE code_units.repo_id = ?3 AND code_units.branch = ?4
                ORDER BY v.distance ASC
                "#,
                params![query_blob, limit as i64, repo_id, branch],
            )
            .await?;
        let mut results = Vec::new();

        while let Some(row) = rows.next().await? {
            let distance = row.get::<f64>(10)? as f32;
            let score = 1.0 / (1.0 + distance.max(0.0));
            results.push(RankedResult {
                unit: code_unit_from_row(&row, 0)
                    .map_err(|error| DatabaseError::InvalidEmbedding(error.to_string()))?,
                fts_score: None,
                vector_score: Some(score),
                combined_score: score,
                match_type: MatchType::Vector,
            });
        }

        Ok(results)
    }

    async fn search_vector_fallback(
        &self,
        repo_id: &str,
        branch: &str,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<RankedResult>> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let mut rows = connection
            .query(
                r#"
                SELECT
                    name,
                    kind,
                    signature,
                    doc,
                    file_path,
                    line_start,
                    line_end,
                    content,
                    parent,
                    language,
                    embedding
                FROM code_units
                WHERE repo_id = ?1 AND branch = ?2 AND embedding IS NOT NULL
                "#,
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;
        let mut results = Vec::new();

        while let Some(row) = rows.next().await.map_err(DatabaseError::from)? {
            let stored_embedding =
                decode_embedding(&row.get::<Vec<u8>>(10).map_err(DatabaseError::from)?)?;
            let score = cosine_similarity(embedding, &stored_embedding)?;
            results.push(RankedResult {
                unit: code_unit_from_row(&row, 0)?,
                fts_score: None,
                vector_score: Some(score),
                combined_score: score,
                match_type: MatchType::Vector,
            });
        }

        results.sort_by(|left, right| {
            right
                .combined_score
                .partial_cmp(&left.combined_score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.unit.name.cmp(&right.unit.name))
        });
        results.truncate(limit);
        Ok(results)
    }

    async fn try_search_hybrid_sql(
        &self,
        repo_id: &str,
        branch: &str,
        sanitized_query: &str,
        embedding: &[f32],
        limit: usize,
        rrf_k: u32,
    ) -> std::result::Result<Vec<RankedResult>, DatabaseError> {
        let query_blob = encode_embedding(embedding)?;
        let candidate_limit = hybrid_candidate_limit(limit);
        let connection = self.connect().await?;
        let mut rows = connection
            .query(
                r#"
                WITH
                fts_source AS (
                    SELECT
                        code_units._rowid AS rowid,
                        bm25(code_units_fts, 6.0, 3.0, 2.0, 1.0) AS score
                    FROM code_units_fts
                    JOIN code_units ON code_units._rowid = code_units_fts.rowid
                    WHERE code_units.repo_id = ?1
                      AND code_units.branch = ?2
                      AND code_units_fts MATCH ?3
                    ORDER BY score ASC, code_units._rowid ASC
                    LIMIT ?5
                ),
                fts_hits AS (
                    SELECT
                        rowid,
                        ROW_NUMBER() OVER (ORDER BY score ASC, rowid ASC) AS rank
                    FROM fts_source
                ),
                vector_source AS (
                    SELECT
                        code_units._rowid AS rowid,
                        v.distance AS distance
                    FROM vector_top_k('idx_code_units_embedding', vector(?4), ?5) AS v
                    JOIN code_units ON code_units._rowid = v.id
                    WHERE code_units.repo_id = ?1
                      AND code_units.branch = ?2
                    ORDER BY v.distance ASC, code_units._rowid ASC
                ),
                vector_hits AS (
                    SELECT
                        rowid,
                        ROW_NUMBER() OVER (ORDER BY distance ASC, rowid ASC) AS rank
                    FROM vector_source
                ),
                unioned AS (
                    SELECT rowid, rank, 1 AS saw_fts, 0 AS saw_vector
                    FROM fts_hits
                    UNION ALL
                    SELECT rowid, rank, 0 AS saw_fts, 1 AS saw_vector
                    FROM vector_hits
                ),
                fused AS (
                    SELECT
                        rowid,
                        SUM(CASE WHEN saw_fts = 1 THEN 1.0 / (?6 + rank) ELSE 0.0 END) AS fts_score,
                        SUM(CASE WHEN saw_vector = 1 THEN 1.0 / (?6 + rank) ELSE 0.0 END) AS vector_score,
                        SUM(1.0 / (?6 + rank)) AS combined_score,
                        MAX(saw_fts) AS saw_fts,
                        MAX(saw_vector) AS saw_vector
                    FROM unioned
                    GROUP BY rowid
                )
                SELECT
                    code_units.name,
                    code_units.kind,
                    code_units.signature,
                    code_units.doc,
                    code_units.file_path,
                    code_units.line_start,
                    code_units.line_end,
                    code_units.content,
                    code_units.parent,
                    code_units.language,
                    CASE WHEN fused.saw_fts = 1 THEN fused.fts_score ELSE NULL END,
                    CASE WHEN fused.saw_vector = 1 THEN fused.vector_score ELSE NULL END,
                    fused.combined_score,
                    CASE
                        WHEN fused.saw_fts = 1 AND fused.saw_vector = 1 THEN 'hybrid'
                        WHEN fused.saw_fts = 1 THEN 'fts'
                        ELSE 'vector'
                    END AS match_type
                FROM fused
                JOIN code_units ON code_units._rowid = fused.rowid
                ORDER BY
                    fused.combined_score DESC,
                    CASE WHEN fused.saw_fts = 1 AND fused.saw_vector = 1 THEN 2 ELSE 1 END DESC,
                    code_units.line_start ASC,
                    code_units.file_path ASC,
                    code_units.name ASC
                LIMIT ?7
                "#,
                params![
                    repo_id,
                    branch,
                    sanitized_query,
                    query_blob,
                    candidate_limit as i64,
                    rrf_k.max(1) as f64,
                    limit as i64,
                ],
            )
            .await?;
        let mut results = Vec::new();

        while let Some(row) = rows.next().await? {
            results.push(ranked_result_from_row(&row, 10, 11, 12, 13)?);
        }

        Ok(results)
    }

    async fn search_hybrid_fallback(
        &self,
        repo_id: &str,
        branch: &str,
        query: &str,
        embedding: &[f32],
        limit: usize,
        rrf_k: u32,
    ) -> Result<Vec<RankedResult>> {
        let candidate_limit = hybrid_candidate_limit(limit);
        let fts_results = self
            .search_fts(repo_id, branch, query, candidate_limit)
            .await?;
        let vector_results = self
            .search_vector(repo_id, branch, embedding, candidate_limit)
            .await
            .unwrap_or_else(|error| {
                warn!(
                    repo_id,
                    branch, query, "vector search unavailable during DB hybrid fallback: {error}"
                );
                Vec::new()
            });

        Ok(fuse_ranked_results(
            fts_results,
            vector_results,
            rrf_k,
            limit,
        ))
    }
}

#[async_trait]
impl Storage for Database {
    async fn store_code_units(
        &self,
        repo_id: &str,
        branch: &str,
        units: &[CodeUnit],
    ) -> Result<()> {
        let indexed_units = units
            .iter()
            .cloned()
            .map(|unit| IndexedCodeUnit {
                file_hash: Some(default_file_hash(&unit)),
                embedding: None,
                unit,
            })
            .collect::<Vec<_>>();
        self.store_indexed_units(repo_id, branch, &indexed_units)
            .await
    }

    async fn search_fts(
        &self,
        repo_id: &str,
        branch: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RankedResult>> {
        let sanitized = match sanitize_fts_query(query) {
            Some(q) => q,
            None => return Ok(Vec::new()),
        };

        let connection = self.connect().await.map_err(TruesightError::from)?;
        let mut rows = connection
            .query(
                r#"
                SELECT
                    code_units.name,
                    code_units.kind,
                    code_units.signature,
                    code_units.doc,
                    code_units.file_path,
                    code_units.line_start,
                    code_units.line_end,
                    code_units.content,
                    code_units.parent,
                    code_units.language,
                    bm25(code_units_fts, 6.0, 3.0, 2.0, 1.0) AS score
                FROM code_units_fts
                JOIN code_units ON code_units._rowid = code_units_fts.rowid
                WHERE code_units.repo_id = ?1
                  AND code_units.branch = ?2
                  AND code_units_fts MATCH ?3
                ORDER BY score ASC, code_units._rowid ASC
                LIMIT ?4
                "#,
                params![repo_id, branch, sanitized, limit as i64],
            )
            .await
            .map_err(DatabaseError::from)?;
        let mut results = Vec::new();

        while let Some(row) = rows.next().await.map_err(DatabaseError::from)? {
            let raw_score = row.get::<f64>(10).map_err(DatabaseError::from)? as f32;
            let normalized = normalize_fts_score(raw_score);
            results.push(RankedResult {
                unit: code_unit_from_row(&row, 0)?,
                fts_score: Some(normalized),
                vector_score: None,
                combined_score: normalized,
                match_type: MatchType::Fts,
            });
        }

        Ok(results)
    }

    async fn search_hybrid(
        &self,
        repo_id: &str,
        branch: &str,
        query: &str,
        embedding: &[f32],
        limit: usize,
        rrf_k: u32,
    ) -> Result<Vec<RankedResult>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        let sanitized = match sanitize_fts_query(query) {
            Some(query) => query,
            None => return self.search_vector(repo_id, branch, embedding, limit).await,
        };
        match self
            .try_search_hybrid_sql(repo_id, branch, &sanitized, embedding, limit, rrf_k)
            .await
        {
            Ok(results) => Ok(results),
            Err(error) => {
                warn!(
                    repo_id,
                    branch,
                    query,
                    "DB-native hybrid search unavailable; using DB fallback fusion: {error}"
                );
                self.search_hybrid_fallback(repo_id, branch, query, embedding, limit, rrf_k)
                    .await
            }
        }
    }

    async fn search_vector(
        &self,
        repo_id: &str,
        branch: &str,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<RankedResult>> {
        if limit == 0 {
            return Ok(Vec::new());
        }

        match self
            .try_vector_top_k(repo_id, branch, embedding, limit)
            .await
        {
            Ok(results) if !results.is_empty() => Ok(results),
            Ok(_) => {
                self.search_vector_fallback(repo_id, branch, embedding, limit)
                    .await
            }
            Err(_) => {
                self.search_vector_fallback(repo_id, branch, embedding, limit)
                    .await
            }
        }
    }

    async fn get_index_metadata(
        &self,
        repo_id: &str,
        branch: &str,
    ) -> Result<Option<IndexMetadata>> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let mut rows = connection
            .query(
                r#"
                SELECT
                    repo_id,
                    branch,
                    status,
                    last_indexed_at,
                    last_commit_sha,
                    last_seen_commit_sha,
                    file_count,
                    symbol_count,
                    embedding_model,
                    last_error
                FROM index_metadata
                WHERE repo_id = ?1 AND branch = ?2
                "#,
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;

        if let Some(row) = rows.next().await.map_err(DatabaseError::from)? {
            return Ok(Some(IndexMetadata {
                repo_id: row.get::<String>(0).map_err(DatabaseError::from)?,
                branch: row.get::<String>(1).map_err(DatabaseError::from)?,
                status: row
                    .get::<String>(2)
                    .map_err(DatabaseError::from)?
                    .parse()
                    .map_err(DatabaseError::InvalidEnumValue)?,
                last_indexed_at: parse_timestamp(
                    &row.get::<String>(3).map_err(DatabaseError::from)?,
                )?,
                last_commit_sha: row.get::<Option<String>>(4).map_err(DatabaseError::from)?,
                last_seen_commit_sha: row.get::<Option<String>>(5).map_err(DatabaseError::from)?,
                file_count: row.get::<i64>(6).map_err(DatabaseError::from)? as u32,
                symbol_count: row.get::<i64>(7).map_err(DatabaseError::from)? as u32,
                embedding_model: row.get::<String>(8).map_err(DatabaseError::from)?,
                last_error: row.get::<Option<String>>(9).map_err(DatabaseError::from)?,
            }));
        }

        Ok(None)
    }

    async fn has_indexed_symbols(&self, repo_id: &str, branch: &str) -> Result<bool> {
        Database::has_indexed_symbols(self, repo_id, branch).await
    }

    async fn set_index_metadata(
        &self,
        repo_id: &str,
        branch: &str,
        meta: &IndexMetadata,
    ) -> Result<()> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        connection
            .execute(
                r#"
                INSERT INTO index_metadata (
                    repo_id,
                    branch,
                    status,
                    last_indexed_at,
                    last_commit_sha,
                    last_seen_commit_sha,
                    file_count,
                    symbol_count,
                    embedding_model,
                    last_error
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
                ON CONFLICT(repo_id, branch) DO UPDATE SET
                    status = excluded.status,
                    last_indexed_at = excluded.last_indexed_at,
                    last_commit_sha = excluded.last_commit_sha,
                    last_seen_commit_sha = excluded.last_seen_commit_sha,
                    file_count = excluded.file_count,
                    symbol_count = excluded.symbol_count,
                    embedding_model = excluded.embedding_model,
                    last_error = excluded.last_error
                "#,
                params![
                    repo_id,
                    branch,
                    meta.status.to_string(),
                    meta.last_indexed_at.to_rfc3339(),
                    meta.last_commit_sha.clone(),
                    meta.last_seen_commit_sha.clone(),
                    i64::from(meta.file_count),
                    i64::from(meta.symbol_count),
                    meta.embedding_model.clone(),
                    meta.last_error.clone(),
                ],
            )
            .await
            .map_err(DatabaseError::from)?;
        Ok(())
    }

    async fn delete_branch_index(&self, repo_id: &str, branch: &str) -> Result<()> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let transaction = connection
            .transaction()
            .await
            .map_err(DatabaseError::from)?;
        transaction
            .execute(
                "DELETE FROM code_units WHERE repo_id = ?1 AND branch = ?2",
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;
        transaction
            .execute(
                "DELETE FROM indexed_files WHERE repo_id = ?1 AND branch = ?2",
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;
        transaction
            .execute(
                "DELETE FROM index_metadata WHERE repo_id = ?1 AND branch = ?2",
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;
        transaction.commit().await.map_err(DatabaseError::from)?;
        Ok(())
    }

    async fn get_all_symbols(&self, repo_id: &str, branch: &str) -> Result<Vec<CodeUnit>> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        let mut rows = connection
            .query(
                r#"
                SELECT name, kind, signature, doc, file_path, line_start, line_end, content, parent, language
                FROM code_units
                WHERE repo_id = ?1 AND branch = ?2
                ORDER BY file_path ASC, line_start ASC, name ASC
                "#,
                params![repo_id, branch],
            )
            .await
            .map_err(DatabaseError::from)?;
        let mut units = Vec::new();

        while let Some(row) = rows.next().await.map_err(DatabaseError::from)? {
            units.push(code_unit_from_row(&row, 0)?);
        }

        Ok(units)
    }
}

#[async_trait]
impl IndexStorage for Database {
    async fn store_indexed_units(
        &self,
        repo_id: &str,
        branch: &str,
        units: &[IndexedCodeUnit],
    ) -> Result<()> {
        Database::store_indexed_units(self, repo_id, branch, units).await
    }

    async fn upsert_indexed_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
        file_hash: &str,
        chunk_count: u32,
    ) -> Result<()> {
        Database::upsert_indexed_file(self, repo_id, branch, file_path, file_hash, chunk_count)
            .await
    }

    async fn list_pending_embeddings(
        &self,
        repo_id: &str,
        branch: &str,
        embedding_model: &str,
        limit: usize,
    ) -> Result<Vec<PendingEmbedding>> {
        Database::list_pending_embeddings(self, repo_id, branch, embedding_model, limit).await
    }

    async fn update_embeddings(
        &self,
        repo_id: &str,
        branch: &str,
        embedding_model: &str,
        updates: &[EmbeddingUpdate],
    ) -> Result<()> {
        Database::update_embeddings(self, repo_id, branch, embedding_model, updates).await
    }

    async fn replace_branch_index(
        &self,
        repo_id: &str,
        branch: &str,
        units: &[IndexedCodeUnit],
        indexed_files: &[IndexedFileRecord],
        metadata: &IndexMetadata,
    ) -> Result<()> {
        Database::replace_branch_index(self, repo_id, branch, units, indexed_files, metadata).await
    }
}

#[async_trait]
impl IncrementalStorage for Database {
    async fn get_indexed_files(
        &self,
        repo_id: &str,
        branch: &str,
    ) -> Result<Vec<IndexedFileRecord>> {
        Database::get_indexed_files(self, repo_id, branch).await
    }

    async fn delete_units_for_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
    ) -> Result<()> {
        Database::delete_units_for_file(self, repo_id, branch, file_path).await
    }

    async fn delete_indexed_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
    ) -> Result<()> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        connection
            .execute(
                "DELETE FROM indexed_files WHERE repo_id = ?1 AND branch = ?2 AND file_path = ?3",
                (repo_id, branch, file_path.to_string_lossy().into_owned()),
            )
            .await
            .map_err(|error| TruesightError::Database(error.to_string()))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        Database, IndexedFileRecord, cosine_similarity, decode_embedding, encode_embedding,
        sanitize_fts_query,
    };
    use chrono::Utc;
    use libsql::params;
    use std::path::{Path, PathBuf};
    use tempfile::TempDir;
    use truesight_core::{
        CodeUnit, CodeUnitKind, IndexMetadata, IndexStatus, IndexedCodeUnit, Language, MatchType,
        Storage,
    };

    const REPO_ID: &str = "/repo";
    const BRANCH: &str = "main";

    async fn open_temp_database() -> (TempDir, Database) {
        let temp_dir = tempfile::tempdir().unwrap();
        let db_path = temp_dir.path().join("storage.db");
        let database = Database::new(&db_path).await.unwrap();
        database.run_migrations().await.unwrap();
        (temp_dir, database)
    }

    async fn sqlite_objects(database: &Database, object_type: &str) -> Vec<String> {
        let connection = database.connect().await.unwrap();
        let mut rows = connection
            .query(
                "SELECT name FROM sqlite_master WHERE type = ?1 ORDER BY name",
                params![object_type],
            )
            .await
            .unwrap();
        let mut names = Vec::new();

        while let Some(row) = rows.next().await.unwrap() {
            names.push(row.get::<String>(0).unwrap());
        }

        names
    }

    async fn migration_rows(database: &Database) -> Vec<(i64, String)> {
        let connection = database.connect().await.unwrap();
        let mut rows = connection
            .query("SELECT version, name FROM _migrations ORDER BY version", ())
            .await
            .unwrap();
        let mut migrations = Vec::new();

        while let Some(row) = rows.next().await.unwrap() {
            migrations.push((row.get::<i64>(0).unwrap(), row.get::<String>(1).unwrap()));
        }

        migrations
    }

    fn sample_unit(name: &str, file_path: &str, line_start: u32, content: &str) -> CodeUnit {
        CodeUnit {
            name: name.to_string(),
            kind: CodeUnitKind::Function,
            signature: format!("pub fn {name}()"),
            doc: Some(format!("Documentation for {name}")),
            file_path: PathBuf::from(file_path),
            line_start,
            line_end: line_start + 2,
            content: content.to_string(),
            parent: None,
            language: Language::Rust,
        }
    }

    fn sample_metadata(
        branch: &str,
        last_commit_sha: Option<&str>,
        file_count: u32,
        symbol_count: u32,
    ) -> IndexMetadata {
        IndexMetadata {
            repo_id: REPO_ID.to_string(),
            branch: branch.to_string(),
            status: IndexStatus::Ready,
            last_indexed_at: Utc::now(),
            last_commit_sha: last_commit_sha.map(ToOwned::to_owned),
            last_seen_commit_sha: last_commit_sha.map(ToOwned::to_owned),
            file_count,
            symbol_count,
            embedding_model: "all-MiniLM-L6-v2".to_string(),
            last_error: None,
        }
    }

    async fn stored_embedding_blob(database: &Database, name: &str) -> Vec<u8> {
        let connection = database.connect().await.unwrap();
        let mut rows = connection
            .query(
                "SELECT embedding FROM code_units WHERE repo_id = ?1 AND branch = ?2 AND name = ?3",
                params![REPO_ID, BRANCH, name],
            )
            .await
            .unwrap();
        rows.next()
            .await
            .unwrap()
            .unwrap()
            .get::<Vec<u8>>(0)
            .unwrap()
    }

    #[tokio::test]
    async fn run_migrations_creates_schema_objects() {
        let (_temp_dir, database) = open_temp_database().await;

        let tables = sqlite_objects(&database, "table").await;
        assert!(tables.contains(&String::from("_migrations")));
        assert!(tables.contains(&String::from("index_metadata")));
        assert!(tables.contains(&String::from("code_units")));
        assert!(tables.contains(&String::from("code_units_fts")));
        assert!(tables.contains(&String::from("indexed_files")));

        let indexes = sqlite_objects(&database, "index").await;
        assert!(indexes.contains(&String::from("idx_code_units_repo_branch")));
        assert!(indexes.contains(&String::from("idx_code_units_file")));
        assert!(indexes.contains(&String::from("idx_code_units_name")));

        let triggers = sqlite_objects(&database, "trigger").await;
        assert!(triggers.contains(&String::from("code_units_fts_insert")));
        assert!(triggers.contains(&String::from("code_units_fts_delete")));
        assert!(triggers.contains(&String::from("code_units_fts_update")));

        let migrations = migration_rows(&database).await;
        assert_eq!(
            migrations,
            vec![
                (1, String::from("initial_schema")),
                (2, String::from("add_index_state_and_embedding_metadata")),
            ]
        );
    }

    #[tokio::test]
    async fn run_migrations_is_idempotent() {
        let (_temp_dir, database) = open_temp_database().await;

        database.run_migrations().await.unwrap();

        let migrations = migration_rows(&database).await;
        assert_eq!(
            migrations,
            vec![
                (1, String::from("initial_schema")),
                (2, String::from("add_index_state_and_embedding_metadata")),
            ]
        );

        let connection = database.connect().await.unwrap();
        let mut rows = connection
            .query(
                "SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'code_units_fts_%'",
                (),
            )
            .await
            .unwrap();
        let row = rows.next().await.unwrap().unwrap();
        let trigger_count = row.get::<i64>(0).unwrap();
        assert_eq!(trigger_count, 3);
    }

    #[test]
    fn db_path_for_repo_is_deterministic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let repo_root = temp_dir.path().join("sample repo");
        std::fs::create_dir_all(&repo_root).unwrap();

        let first_path = Database::db_path_for_repo(&repo_root).unwrap();
        let second_path = Database::db_path_for_repo(&repo_root).unwrap();
        let other_path = Database::db_path_for_repo(&temp_dir.path().join("another-repo")).unwrap();

        assert_eq!(first_path, second_path);
        assert_ne!(first_path, other_path);
        assert_eq!(
            first_path.extension().and_then(|ext| ext.to_str()),
            Some("db")
        );

        let file_name = first_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap();
        assert!(file_name.starts_with("sample-repo-"));
    }

    #[test]
    fn config_dir_uses_truesight_directory() {
        let config_dir = Database::config_dir().unwrap();

        assert!(config_dir.ends_with(".config/truesight"));
        assert!(config_dir.exists());
    }

    #[tokio::test]
    async fn store_and_get_all_symbols_round_trips_units() {
        let (_temp_dir, database) = open_temp_database().await;
        let alpha = sample_unit(
            "alpha_handler",
            "src/alpha.rs",
            10,
            "pub fn alpha_handler() -> bool { true }",
        );
        let beta = sample_unit(
            "beta_handler",
            "src/beta.rs",
            20,
            "pub fn beta_handler() -> bool { false }",
        );
        let gamma = sample_unit(
            "gamma_handler",
            "src/gamma.rs",
            30,
            "pub fn gamma_handler() -> bool { alpha_handler() }",
        );

        database
            .store_code_units(
                REPO_ID,
                BRANCH,
                &[alpha.clone(), beta.clone(), gamma.clone()],
            )
            .await
            .unwrap();

        let stored = database.get_all_symbols(REPO_ID, BRANCH).await.unwrap();

        assert_eq!(stored.len(), 3);
        assert_eq!(stored[0].name, "alpha_handler");
        assert_eq!(stored[1].name, "beta_handler");
        assert_eq!(stored[2].name, "gamma_handler");
        assert_eq!(stored[0].signature, alpha.signature);
        assert_eq!(stored[1].file_path, beta.file_path);
        assert_eq!(stored[2].content, gamma.content);
    }

    #[tokio::test]
    async fn search_fts_returns_ranked_match_from_fts_triggers() {
        let (_temp_dir, database) = open_temp_database().await;
        let alpha = sample_unit(
            "alpha_handler",
            "src/alpha.rs",
            10,
            "pub fn alpha_handler() -> bool { true }",
        );
        let target = sample_unit(
            "token_refresh",
            "src/token.rs",
            20,
            "pub fn token_refresh() -> bool { validate_token_refresh() }",
        );
        let gamma = sample_unit(
            "gamma_handler",
            "src/gamma.rs",
            30,
            "pub fn gamma_handler() -> bool { false }",
        );

        database
            .store_code_units(REPO_ID, BRANCH, &[alpha, target.clone(), gamma])
            .await
            .unwrap();

        let results = database
            .search_fts(REPO_ID, BRANCH, "token_refresh", 5)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].unit.name, target.name);
        assert_eq!(results[0].unit.file_path, target.file_path);
        assert_eq!(results[0].match_type, MatchType::Fts);
        assert!(results[0].fts_score.unwrap() > 0.0);

        let empty = database
            .search_fts(REPO_ID, BRANCH, "missing_symbol", 5)
            .await
            .unwrap();
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn search_fts_weights_name_and_signature_above_content_only_matches() {
        let (_temp_dir, database) = open_temp_database().await;
        let content_only = sample_unit(
            "helper",
            "src/helper.rs",
            10,
            "pub fn helper() -> bool { let note = \"token_refresh token_refresh\"; true }",
        );
        let exact_symbol = sample_unit(
            "token_refresh",
            "src/token.rs",
            20,
            "pub fn token_refresh() -> bool { true }",
        );

        database
            .store_code_units(REPO_ID, BRANCH, &[content_only, exact_symbol])
            .await
            .unwrap();

        let results = database
            .search_fts(REPO_ID, BRANCH, "token_refresh", 2)
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].unit.name, "token_refresh");
        assert_eq!(results[1].unit.name, "helper");
    }

    #[test]
    fn sanitize_fts_query_quotes_each_token() {
        assert_eq!(
            sanitize_fts_query("hello world"),
            Some(r#""hello" "world""#.to_string())
        );
    }

    #[test]
    fn sanitize_fts_query_escapes_internal_double_quotes() {
        // Input: say "hello"
        // Tokens after split_whitespace: ["say", "\"hello\""]
        // "hello" → replace " with "": ""hello"" → wrap: """"hello""""
        assert_eq!(
            sanitize_fts_query(r#"say "hello""#),
            Some("\"say\" \"\"\"hello\"\"\"".to_string())
        );
    }

    #[test]
    fn sanitize_fts_query_handles_question_mark() {
        assert_eq!(
            sanitize_fts_query("what languages are supported?"),
            Some(r#""what" "languages" "are" "supported?""#.to_string())
        );
    }

    #[test]
    fn sanitize_fts_query_handles_fts_operators() {
        assert_eq!(
            sanitize_fts_query("foo* (bar) -baz"),
            Some(r#""foo*" "(bar)" "-baz""#.to_string())
        );
    }

    #[test]
    fn sanitize_fts_query_returns_none_for_empty_input() {
        assert_eq!(sanitize_fts_query(""), None);
        assert_eq!(sanitize_fts_query("   "), None);
    }

    #[tokio::test]
    async fn search_fts_handles_special_characters_without_error() {
        let (_temp_dir, database) = open_temp_database().await;
        let unit = sample_unit(
            "supported_languages",
            "src/lang.rs",
            10,
            "pub fn supported_languages() -> Vec<String> { vec![] }",
        );

        database
            .store_code_units(REPO_ID, BRANCH, &[unit])
            .await
            .unwrap();

        let results = database
            .search_fts(REPO_ID, BRANCH, "supported?", 5)
            .await
            .unwrap();
        assert!(!results.is_empty());
        assert_eq!(results[0].unit.name, "supported_languages");

        let results = database
            .search_fts(REPO_ID, BRANCH, r#""supported""#, 5)
            .await
            .unwrap();
        assert!(!results.is_empty());

        let results = database
            .search_fts(REPO_ID, BRANCH, "languages (supported)", 5)
            .await
            .unwrap();
        assert!(!results.is_empty());

        let results = database.search_fts(REPO_ID, BRANCH, "", 5).await.unwrap();
        assert!(results.is_empty());

        let results = database
            .search_fts(REPO_ID, BRANCH, "   ", 5)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_vector_uses_blob_embeddings_and_returns_nearest_neighbor() {
        let (_temp_dir, database) = open_temp_database().await;
        let alpha = IndexedCodeUnit {
            unit: sample_unit(
                "alpha_handler",
                "src/alpha.rs",
                10,
                "pub fn alpha_handler() -> bool { true }",
            ),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            file_hash: None,
        };
        let beta = IndexedCodeUnit {
            unit: sample_unit(
                "beta_handler",
                "src/beta.rs",
                20,
                "pub fn beta_handler() -> bool { false }",
            ),
            embedding: Some(vec![0.0, 1.0, 0.0, 0.0]),
            file_hash: None,
        };
        let gamma = IndexedCodeUnit {
            unit: sample_unit(
                "gamma_handler",
                "src/gamma.rs",
                30,
                "pub fn gamma_handler() -> bool { false }",
            ),
            embedding: Some(vec![0.8, 0.2, 0.0, 0.0]),
            file_hash: None,
        };

        database
            .store_indexed_units(REPO_ID, BRANCH, &[alpha, beta, gamma])
            .await
            .unwrap();

        let blob = stored_embedding_blob(&database, "alpha_handler").await;
        assert_eq!(blob.len(), 16);
        assert_eq!(decode_embedding(&blob).unwrap(), vec![1.0, 0.0, 0.0, 0.0]);

        let results = database
            .search_vector(REPO_ID, BRANCH, &[1.0, 0.0, 0.0, 0.0], 3)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].unit.name, "alpha_handler");
        assert!(results[0].vector_score.unwrap() > results[1].vector_score.unwrap());
        assert_eq!(results[0].match_type, MatchType::Vector);

        let empty = database
            .search_vector(REPO_ID, BRANCH, &[0.0, 0.0, 1.0, 0.0], 0)
            .await
            .unwrap();
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn search_hybrid_fuses_lexical_and_vector_hits() {
        let (_temp_dir, database) = open_temp_database().await;
        let hybrid = IndexedCodeUnit {
            unit: sample_unit(
                "authenticate_user",
                "src/auth.rs",
                10,
                "pub fn authenticate_user() -> bool { true }",
            ),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            file_hash: None,
        };
        let lexical_only = IndexedCodeUnit {
            unit: sample_unit(
                "user_validator",
                "src/user.rs",
                20,
                "pub fn user_validator() -> bool { let label = \"authenticate user\"; true }",
            ),
            embedding: Some(vec![0.0, 1.0, 0.0, 0.0]),
            file_hash: None,
        };
        let vector_only = IndexedCodeUnit {
            unit: sample_unit(
                "semantic_auth",
                "src/semantic.rs",
                30,
                "pub fn semantic_auth() -> bool { false }",
            ),
            embedding: Some(vec![0.95, 0.05, 0.0, 0.0]),
            file_hash: None,
        };

        database
            .store_indexed_units(REPO_ID, BRANCH, &[hybrid, lexical_only, vector_only])
            .await
            .unwrap();

        let results = database
            .search_hybrid(
                REPO_ID,
                BRANCH,
                "authenticate user",
                &[1.0, 0.0, 0.0, 0.0],
                3,
                60,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].unit.name, "authenticate_user");
        assert_eq!(results[0].match_type, MatchType::Hybrid);
        assert!(
            results
                .iter()
                .any(|result| result.match_type == MatchType::Vector)
        );
        assert!(results[0].combined_score > results[1].combined_score);
    }

    #[tokio::test]
    async fn search_hybrid_uses_vector_results_when_query_has_no_fts_tokens() {
        let (_temp_dir, database) = open_temp_database().await;
        let target = IndexedCodeUnit {
            unit: sample_unit(
                "semantic_auth",
                "src/semantic.rs",
                10,
                "pub fn semantic_auth() -> bool { true }",
            ),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            file_hash: None,
        };

        database
            .store_indexed_units(REPO_ID, BRANCH, &[target])
            .await
            .unwrap();

        let results = database
            .search_hybrid(REPO_ID, BRANCH, "   ", &[1.0, 0.0, 0.0, 0.0], 1, 60)
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].unit.name, "semantic_auth");
        assert_eq!(results[0].match_type, MatchType::Vector);
    }

    #[tokio::test]
    async fn search_hybrid_fallback_tracks_source_specific_rrf_scores() {
        let (_temp_dir, database) = open_temp_database().await;
        let hybrid = IndexedCodeUnit {
            unit: sample_unit(
                "authenticate_user",
                "src/auth.rs",
                10,
                "pub fn authenticate_user() -> bool { true }",
            ),
            embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
            file_hash: None,
        };
        let vector_only = IndexedCodeUnit {
            unit: sample_unit(
                "semantic_auth",
                "src/semantic.rs",
                30,
                "pub fn semantic_auth() -> bool { false }",
            ),
            embedding: Some(vec![0.95, 0.05, 0.0, 0.0]),
            file_hash: None,
        };

        database
            .store_indexed_units(REPO_ID, BRANCH, &[hybrid, vector_only])
            .await
            .unwrap();

        let results = database
            .search_hybrid_fallback(
                REPO_ID,
                BRANCH,
                "authenticate user",
                &[1.0, 0.0, 0.0, 0.0],
                2,
                60,
            )
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].unit.name, "authenticate_user");
        assert_eq!(results[0].match_type, MatchType::Hybrid);
        assert!(results[0].fts_score.unwrap() > 0.0);
        assert!(results[0].vector_score.unwrap() > 0.0);
        assert!(results[0].combined_score > results[0].fts_score.unwrap());
        assert!(results[0].combined_score > results[0].vector_score.unwrap());
    }

    #[tokio::test]
    async fn delete_units_for_file_removes_only_requested_file() {
        let (_temp_dir, database) = open_temp_database().await;
        let alpha = sample_unit(
            "alpha_handler",
            "src/alpha.rs",
            10,
            "pub fn alpha_handler() -> bool { true }",
        );
        let alpha_helper = sample_unit(
            "alpha_helper",
            "src/alpha.rs",
            14,
            "pub fn alpha_helper() -> bool { true }",
        );
        let beta = sample_unit(
            "beta_handler",
            "src/beta.rs",
            20,
            "pub fn beta_handler() -> bool { false }",
        );

        database
            .store_code_units(REPO_ID, BRANCH, &[alpha, alpha_helper, beta])
            .await
            .unwrap();

        database
            .delete_units_for_file(REPO_ID, BRANCH, Path::new("src/alpha.rs"))
            .await
            .unwrap();

        let stored = database.get_all_symbols(REPO_ID, BRANCH).await.unwrap();
        assert_eq!(stored.len(), 1);
        assert_eq!(stored[0].name, "beta_handler");
    }

    #[tokio::test]
    async fn index_metadata_round_trips() {
        let (_temp_dir, database) = open_temp_database().await;
        let metadata = sample_metadata(BRANCH, Some("abc123"), 7, 21);

        database
            .set_index_metadata(REPO_ID, BRANCH, &metadata)
            .await
            .unwrap();

        let stored = database
            .get_index_metadata(REPO_ID, BRANCH)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.repo_id, metadata.repo_id);
        assert_eq!(stored.branch, metadata.branch);
        assert_eq!(stored.last_commit_sha, metadata.last_commit_sha);
        assert_eq!(stored.file_count, metadata.file_count);
        assert_eq!(stored.symbol_count, metadata.symbol_count);

        let missing = database
            .get_index_metadata(REPO_ID, "feature")
            .await
            .unwrap();
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn indexed_files_round_trip_and_branch_delete_cleans_related_rows() {
        let (_temp_dir, database) = open_temp_database().await;
        database
            .upsert_indexed_file(REPO_ID, BRANCH, Path::new("src/alpha.rs"), "hash-alpha", 1)
            .await
            .unwrap();
        database
            .upsert_indexed_file(REPO_ID, BRANCH, Path::new("src/beta.rs"), "hash-beta", 2)
            .await
            .unwrap();
        database
            .set_index_metadata(REPO_ID, BRANCH, &sample_metadata(BRANCH, None, 2, 3))
            .await
            .unwrap();
        database
            .store_code_units(
                REPO_ID,
                BRANCH,
                &[sample_unit(
                    "alpha_handler",
                    "src/alpha.rs",
                    10,
                    "pub fn alpha_handler() -> bool { true }",
                )],
            )
            .await
            .unwrap();

        let files = database.get_indexed_files(REPO_ID, BRANCH).await.unwrap();
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].file_path, PathBuf::from("src/alpha.rs"));
        assert_eq!(files[0].file_hash, "hash-alpha");
        assert!(matches!(files[0], IndexedFileRecord { .. }));

        database.delete_branch_index(REPO_ID, BRANCH).await.unwrap();

        assert!(
            database
                .get_all_symbols(REPO_ID, BRANCH)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            database
                .get_indexed_files(REPO_ID, BRANCH)
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            database
                .get_index_metadata(REPO_ID, BRANCH)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn list_branches_returns_distinct_sorted_branch_names() {
        let (_temp_dir, database) = open_temp_database().await;

        for branch in ["feature/auth", "main", "feature/auth", "detached-deadbeef"] {
            database
                .set_index_metadata(REPO_ID, branch, &sample_metadata(branch, None, 0, 0))
                .await
                .unwrap();
        }

        let branches = database.list_branches(REPO_ID).await.unwrap();

        assert_eq!(
            branches,
            vec![
                "detached-deadbeef".to_string(),
                "feature/auth".to_string(),
                "main".to_string(),
            ]
        );
    }

    #[tokio::test]
    async fn branch_scoped_queries_keep_metadata_and_symbols_isolated() {
        let (_temp_dir, database) = open_temp_database().await;
        let main_unit = sample_unit(
            "main_only",
            "src/lib.rs",
            10,
            "pub fn main_only() -> bool { true }",
        );
        let feature_unit = sample_unit(
            "feature_only",
            "src/lib.rs",
            10,
            "pub fn feature_only() -> bool { true }",
        );

        database
            .store_code_units(REPO_ID, "main", std::slice::from_ref(&main_unit))
            .await
            .unwrap();
        database
            .store_code_units(REPO_ID, "feature/auth", std::slice::from_ref(&feature_unit))
            .await
            .unwrap();
        database
            .set_index_metadata(
                REPO_ID,
                "main",
                &sample_metadata("main", Some("1111111"), 1, 1),
            )
            .await
            .unwrap();
        database
            .set_index_metadata(
                REPO_ID,
                "feature/auth",
                &sample_metadata("feature/auth", Some("2222222"), 1, 1),
            )
            .await
            .unwrap();

        let main_symbols = database.get_all_symbols(REPO_ID, "main").await.unwrap();
        let feature_symbols = database
            .get_all_symbols(REPO_ID, "feature/auth")
            .await
            .unwrap();
        let main_search = database
            .search_fts(REPO_ID, "main", "main_only", 5)
            .await
            .unwrap();
        let feature_search = database
            .search_fts(REPO_ID, "feature/auth", "feature_only", 5)
            .await
            .unwrap();
        let missing_cross_branch = database
            .search_fts(REPO_ID, "main", "feature_only", 5)
            .await
            .unwrap();

        assert_eq!(main_symbols.len(), 1);
        assert_eq!(feature_symbols.len(), 1);
        assert_eq!(main_symbols[0].name, main_unit.name);
        assert_eq!(feature_symbols[0].name, feature_unit.name);
        assert_eq!(main_search.len(), 1);
        assert_eq!(main_search[0].unit.name, "main_only");
        assert_eq!(feature_search.len(), 1);
        assert_eq!(feature_search[0].unit.name, "feature_only");
        assert!(missing_cross_branch.is_empty());
        assert_eq!(
            database
                .get_index_metadata(REPO_ID, "main")
                .await
                .unwrap()
                .unwrap()
                .last_commit_sha,
            Some("1111111".to_string())
        );
        assert_eq!(
            database
                .get_index_metadata(REPO_ID, "feature/auth")
                .await
                .unwrap()
                .unwrap()
                .last_commit_sha,
            Some("2222222".to_string())
        );
    }

    #[tokio::test]
    async fn cleanup_branch_removes_only_requested_partition() {
        let (_temp_dir, database) = open_temp_database().await;

        database
            .store_code_units(
                REPO_ID,
                "main",
                &[sample_unit(
                    "main_only",
                    "src/lib.rs",
                    10,
                    "pub fn main_only() -> bool { true }",
                )],
            )
            .await
            .unwrap();
        database
            .store_code_units(
                REPO_ID,
                "feature/auth",
                &[sample_unit(
                    "feature_only",
                    "src/lib.rs",
                    10,
                    "pub fn feature_only() -> bool { true }",
                )],
            )
            .await
            .unwrap();
        database
            .upsert_indexed_file(REPO_ID, "main", Path::new("src/lib.rs"), "hash-main", 1)
            .await
            .unwrap();
        database
            .upsert_indexed_file(
                REPO_ID,
                "feature/auth",
                Path::new("src/lib.rs"),
                "hash-feature",
                1,
            )
            .await
            .unwrap();
        database
            .set_index_metadata(REPO_ID, "main", &sample_metadata("main", None, 1, 1))
            .await
            .unwrap();
        database
            .set_index_metadata(
                REPO_ID,
                "feature/auth",
                &sample_metadata("feature/auth", None, 1, 1),
            )
            .await
            .unwrap();

        database
            .cleanup_branch(REPO_ID, "feature/auth")
            .await
            .unwrap();

        assert_eq!(
            database
                .get_all_symbols(REPO_ID, "main")
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            database
                .get_indexed_files(REPO_ID, "main")
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(
            database
                .get_index_metadata(REPO_ID, "main")
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            database
                .get_all_symbols(REPO_ID, "feature/auth")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            database
                .get_indexed_files(REPO_ID, "feature/auth")
                .await
                .unwrap()
                .is_empty()
        );
        assert!(
            database
                .get_index_metadata(REPO_ID, "feature/auth")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn embedding_helpers_round_trip_and_reject_invalid_inputs() {
        let blob = encode_embedding(&[1.0, 2.5, -3.0, 4.25]).unwrap();
        assert_eq!(decode_embedding(&blob).unwrap(), vec![1.0, 2.5, -3.0, 4.25]);

        let invalid = decode_embedding(&[1, 2, 3]).unwrap_err();
        assert!(
            invalid
                .to_string()
                .contains("invalid embedding blob length")
        );
        assert!(encode_embedding(&[]).is_err());
    }

    #[test]
    fn cosine_similarity_handles_exact_and_mismatched_vectors() {
        let exact = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
        assert!((exact - 1.0).abs() < f32::EPSILON);

        let orthogonal = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(orthogonal.abs() < f32::EPSILON);

        let mismatched = cosine_similarity(&[1.0], &[1.0, 0.0]).unwrap_err();
        assert!(
            mismatched
                .to_string()
                .contains("embedding dimension mismatch")
        );
    }
}
