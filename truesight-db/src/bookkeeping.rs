use super::*;

pub(super) const UPSERT_INDEXED_FILE_SQL: &str = r#"
    INSERT INTO indexed_files (repo_id, branch, file_path, file_hash, chunk_count)
    VALUES (?1, ?2, ?3, ?4, ?5)
    ON CONFLICT(repo_id, branch, file_path) DO UPDATE SET
        file_hash = excluded.file_hash,
        chunk_count = excluded.chunk_count,
        indexed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
"#;

pub(super) const UPSERT_INDEX_METADATA_SQL: &str = r#"
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
"#;

impl Database {
    pub async fn apply_incremental_changes(
        &self,
        repo_id: &str,
        branch: &str,
        deleted_files: &[PathBuf],
        units: &[IndexedCodeUnit],
        indexed_files: &[IndexedFileRecord],
    ) -> Result<()> {
        if deleted_files.is_empty() && units.is_empty() && indexed_files.is_empty() {
            return Ok(());
        }

        let connection = self.connect().await.map_err(TruesightError::from)?;
        let transaction = connection
            .transaction()
            .await
            .map_err(DatabaseError::from)?;

        for file_path in deleted_files {
            delete_file_partition(&transaction, repo_id, branch, file_path).await?;
        }

        for entry in units {
            insert_code_unit(&transaction, repo_id, branch, entry)
                .await
                .map_err(TruesightError::from)?;
        }

        for file in indexed_files {
            upsert_indexed_file_in_transaction(
                &transaction,
                repo_id,
                branch,
                &file.file_path,
                &file.file_hash,
                file.chunk_count,
            )
            .await?;
        }

        transaction.commit().await.map_err(DatabaseError::from)?;
        Ok(())
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
                UPSERT_INDEXED_FILE_SQL,
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

        delete_branch_partition(&transaction, repo_id, branch)
            .await
            .map_err(TruesightError::from)?;

        for entry in units {
            insert_code_unit(&transaction, repo_id, branch, entry)
                .await
                .map_err(TruesightError::from)?;
        }

        for file in indexed_files {
            upsert_indexed_file_in_transaction(
                &transaction,
                repo_id,
                branch,
                &file.file_path,
                &file.file_hash,
                file.chunk_count,
            )
            .await?;
        }

        upsert_index_metadata_record(&transaction, repo_id, branch, metadata).await?;

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
    pub async fn list_branches(&self, repo_id: &str) -> Result<Vec<String>> {
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
    pub async fn cleanup_branch(&self, repo_id: &str, branch: &str) -> Result<()> {
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
}

pub(super) async fn delete_branch_partition(
    transaction: &libsql::Transaction,
    repo_id: &str,
    branch: &str,
) -> std::result::Result<(), DatabaseError> {
    transaction
        .execute(
            "DELETE FROM code_units WHERE repo_id = ?1 AND branch = ?2",
            params![repo_id, branch],
        )
        .await?;
    transaction
        .execute(
            "DELETE FROM indexed_files WHERE repo_id = ?1 AND branch = ?2",
            params![repo_id, branch],
        )
        .await?;
    transaction
        .execute(
            "DELETE FROM index_metadata WHERE repo_id = ?1 AND branch = ?2",
            params![repo_id, branch],
        )
        .await?;
    Ok(())
}

pub(super) async fn delete_file_partition(
    transaction: &libsql::Transaction,
    repo_id: &str,
    branch: &str,
    file_path: &Path,
) -> std::result::Result<(), DatabaseError> {
    let file_path = path_to_string(file_path);

    transaction
        .execute(
            "DELETE FROM code_units WHERE repo_id = ?1 AND branch = ?2 AND file_path = ?3",
            params![repo_id, branch, file_path.clone()],
        )
        .await?;
    transaction
        .execute(
            "DELETE FROM indexed_files WHERE repo_id = ?1 AND branch = ?2 AND file_path = ?3",
            params![repo_id, branch, file_path],
        )
        .await?;
    Ok(())
}

pub(super) async fn upsert_indexed_file_in_transaction(
    transaction: &libsql::Transaction,
    repo_id: &str,
    branch: &str,
    file_path: &Path,
    file_hash: &str,
    chunk_count: u32,
) -> std::result::Result<(), DatabaseError> {
    transaction
        .execute(
            UPSERT_INDEXED_FILE_SQL,
            params![
                repo_id,
                branch,
                path_to_string(file_path),
                file_hash,
                i64::from(chunk_count),
            ],
        )
        .await?;
    Ok(())
}

pub(super) async fn upsert_index_metadata_record(
    transaction: &libsql::Transaction,
    repo_id: &str,
    branch: &str,
    metadata: &IndexMetadata,
) -> std::result::Result<(), DatabaseError> {
    transaction
        .execute(
            UPSERT_INDEX_METADATA_SQL,
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
        .await?;
    Ok(())
}
