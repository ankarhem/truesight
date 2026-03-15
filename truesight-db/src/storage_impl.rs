use super::*;
use crate::bookkeeping::{UPSERT_INDEX_METADATA_SQL, delete_branch_partition};

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
                UPSERT_INDEX_METADATA_SQL,
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
        delete_branch_partition(&transaction, repo_id, branch)
            .await
            .map_err(TruesightError::from)?;
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

    async fn apply_incremental_changes(
        &self,
        repo_id: &str,
        branch: &str,
        deleted_files: &[PathBuf],
        units: &[IndexedCodeUnit],
        indexed_files: &[IndexedFileRecord],
    ) -> Result<()> {
        Database::apply_incremental_changes(
            self,
            repo_id,
            branch,
            deleted_files,
            units,
            indexed_files,
        )
        .await
    }
}
