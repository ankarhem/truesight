use super::*;

impl Database {
    pub async fn try_vector_top_k(
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

    pub async fn search_vector_fallback(
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

    pub async fn try_search_hybrid_sql(
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

    pub async fn search_hybrid_fallback(
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
