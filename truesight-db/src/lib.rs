use std::cmp::Ordering;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use libsql::{Connection, Database as LibsqlDatabase, params};
use sha2::{Digest, Sha256};
use truesight_core::{
    CodeUnit, CodeUnitKind, IncrementalStorage, IndexMetadata, IndexStorage, IndexedCodeUnit,
    IndexedFileRecord, Language, MatchType, RankedResult, Result, Storage, TruesightError,
};

const MIGRATION_TABLE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS _migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"#;

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS index_metadata (
    repo_id TEXT NOT NULL,
    branch TEXT NOT NULL,
    last_indexed_at TEXT NOT NULL,
    last_commit_sha TEXT,
    file_count INTEGER NOT NULL DEFAULT 0,
    symbol_count INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (repo_id, branch)
);

CREATE TABLE IF NOT EXISTS code_units (
    _rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    id TEXT NOT NULL UNIQUE,
    repo_id TEXT NOT NULL,
    branch TEXT NOT NULL,
    file_path TEXT NOT NULL,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    signature TEXT NOT NULL,
    doc TEXT,
    content TEXT NOT NULL,
    parent TEXT,
    language TEXT NOT NULL,
    line_start INTEGER NOT NULL,
    line_end INTEGER NOT NULL,
    embedding BLOB,
    file_hash TEXT NOT NULL,
    UNIQUE (repo_id, branch, file_path, name, kind, line_start)
);

CREATE INDEX IF NOT EXISTS idx_code_units_repo_branch ON code_units(repo_id, branch);
CREATE INDEX IF NOT EXISTS idx_code_units_file ON code_units(repo_id, branch, file_path);
CREATE INDEX IF NOT EXISTS idx_code_units_name ON code_units(name);

CREATE VIRTUAL TABLE IF NOT EXISTS code_units_fts USING fts5(
    name,
    signature,
    doc,
    content,
    content='code_units',
    content_rowid='_rowid'
);

CREATE TRIGGER IF NOT EXISTS code_units_fts_insert AFTER INSERT ON code_units BEGIN
    INSERT INTO code_units_fts(rowid, name, signature, doc, content)
    VALUES (new._rowid, new.name, new.signature, new.doc, new.content);
END;

CREATE TRIGGER IF NOT EXISTS code_units_fts_delete AFTER DELETE ON code_units BEGIN
    INSERT INTO code_units_fts(code_units_fts, rowid, name, signature, doc, content)
    VALUES ('delete', old._rowid, old.name, old.signature, old.doc, old.content);
END;

CREATE TRIGGER IF NOT EXISTS code_units_fts_update AFTER UPDATE ON code_units BEGIN
    INSERT INTO code_units_fts(code_units_fts, rowid, name, signature, doc, content)
    VALUES ('delete', old._rowid, old.name, old.signature, old.doc, old.content);
    INSERT INTO code_units_fts(rowid, name, signature, doc, content)
    VALUES (new._rowid, new.name, new.signature, new.doc, new.content);
END;

CREATE TABLE IF NOT EXISTS indexed_files (
    repo_id TEXT NOT NULL,
    branch TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL,
    indexed_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (repo_id, branch, file_path)
);
"#;

const MIGRATIONS: &[(i64, &str, &str)] = &[(1, "initial_schema", SCHEMA)];

#[derive(Debug)]
pub enum DatabaseError {
    Io(std::io::Error),
    Libsql(libsql::Error),
    MissingHomeDirectory,
    InvalidTimestamp(String),
    InvalidEmbeddingLength(usize),
    InvalidEmbedding(String),
}

impl std::fmt::Display for DatabaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "io error: {error}"),
            Self::Libsql(error) => write!(f, "database error: {error}"),
            Self::MissingHomeDirectory => write!(f, "HOME is not set"),
            Self::InvalidTimestamp(value) => write!(f, "invalid timestamp: {value}"),
            Self::InvalidEmbeddingLength(length) => {
                write!(f, "invalid embedding blob length: {length}")
            }
            Self::InvalidEmbedding(message) => write!(f, "invalid embedding: {message}"),
        }
    }
}

impl std::error::Error for DatabaseError {}

impl From<std::io::Error> for DatabaseError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<libsql::Error> for DatabaseError {
    fn from(value: libsql::Error) -> Self {
        Self::Libsql(value)
    }
}

impl From<DatabaseError> for TruesightError {
    fn from(value: DatabaseError) -> Self {
        TruesightError::Database(value.to_string())
    }
}

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
        run_incremental_migrations(&connection).await
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
            let unit_id = code_unit_id(repo_id, branch, &entry.unit);
            let embedding_blob = entry
                .embedding
                .as_deref()
                .map(encode_embedding)
                .transpose()
                .map_err(TruesightError::from)?;
            let file_hash = entry
                .file_hash
                .clone()
                .unwrap_or_else(|| default_file_hash(&entry.unit));

            transaction
                .execute(
                    r#"
                    INSERT INTO code_units (
                        id,
                        repo_id,
                        branch,
                        file_path,
                        name,
                        kind,
                        signature,
                        doc,
                        content,
                        parent,
                        language,
                        line_start,
                        line_end,
                        embedding,
                        file_hash
                    )
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
                    ON CONFLICT(repo_id, branch, file_path, name, kind, line_start) DO UPDATE SET
                        id = excluded.id,
                        signature = excluded.signature,
                        doc = excluded.doc,
                        content = excluded.content,
                        parent = excluded.parent,
                        language = excluded.language,
                        line_end = excluded.line_end,
                        embedding = excluded.embedding,
                        file_hash = excluded.file_hash
                    "#,
                    params![
                        unit_id,
                        repo_id,
                        branch,
                        path_to_string(&entry.unit.file_path),
                        entry.unit.name.clone(),
                        code_unit_kind_to_str(entry.unit.kind),
                        entry.unit.signature.clone(),
                        entry.unit.doc.clone(),
                        entry.unit.content.clone(),
                        entry.unit.parent.clone(),
                        language_to_str(entry.unit.language),
                        i64::from(entry.unit.line_start),
                        i64::from(entry.unit.line_end),
                        embedding_blob,
                        file_hash,
                    ],
                )
                .await
                .map_err(DatabaseError::from)?;
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
    ) -> Result<()> {
        let connection = self.connect().await.map_err(TruesightError::from)?;
        connection
            .execute(
                r#"
                INSERT INTO indexed_files (repo_id, branch, file_path, file_hash)
                VALUES (?1, ?2, ?3, ?4)
                ON CONFLICT(repo_id, branch, file_path) DO UPDATE SET
                    file_hash = excluded.file_hash,
                    indexed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                "#,
                params![repo_id, branch, path_to_string(file_path), file_hash],
            )
            .await
            .map_err(DatabaseError::from)?;
        Ok(())
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
            let unit_id = code_unit_id(repo_id, branch, &entry.unit);
            let embedding_blob = entry
                .embedding
                .as_deref()
                .map(encode_embedding)
                .transpose()
                .map_err(TruesightError::from)?;
            let file_hash = entry
                .file_hash
                .clone()
                .unwrap_or_else(|| default_file_hash(&entry.unit));

            transaction
                .execute(
                    r#"
                    INSERT INTO code_units (
                        id,
                        repo_id,
                        branch,
                        file_path,
                        name,
                        kind,
                        signature,
                        doc,
                        content,
                        parent,
                        language,
                        line_start,
                        line_end,
                        embedding,
                        file_hash
                    )
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)
                    ON CONFLICT(repo_id, branch, file_path, name, kind, line_start) DO UPDATE SET
                        id = excluded.id,
                        signature = excluded.signature,
                        doc = excluded.doc,
                        content = excluded.content,
                        parent = excluded.parent,
                        language = excluded.language,
                        line_end = excluded.line_end,
                        embedding = excluded.embedding,
                        file_hash = excluded.file_hash
                    "#,
                    params![
                        unit_id,
                        repo_id,
                        branch,
                        path_to_string(&entry.unit.file_path),
                        entry.unit.name.clone(),
                        code_unit_kind_to_str(entry.unit.kind),
                        entry.unit.signature.clone(),
                        entry.unit.doc.clone(),
                        entry.unit.content.clone(),
                        entry.unit.parent.clone(),
                        language_to_str(entry.unit.language),
                        i64::from(entry.unit.line_start),
                        i64::from(entry.unit.line_end),
                        embedding_blob,
                        file_hash,
                    ],
                )
                .await
                .map_err(DatabaseError::from)?;
        }

        for file in indexed_files {
            transaction
                .execute(
                    r#"
                    INSERT INTO indexed_files (repo_id, branch, file_path, file_hash)
                    VALUES (?1, ?2, ?3, ?4)
                    ON CONFLICT(repo_id, branch, file_path) DO UPDATE SET
                        file_hash = excluded.file_hash,
                        indexed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    "#,
                    params![
                        repo_id,
                        branch,
                        path_to_string(&file.file_path),
                        file.file_hash.clone()
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
                    last_indexed_at,
                    last_commit_sha,
                    file_count,
                    symbol_count
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ON CONFLICT(repo_id, branch) DO UPDATE SET
                    last_indexed_at = excluded.last_indexed_at,
                    last_commit_sha = excluded.last_commit_sha,
                    file_count = excluded.file_count,
                    symbol_count = excluded.symbol_count
                "#,
                params![
                    repo_id,
                    branch,
                    metadata.last_indexed_at.to_rfc3339(),
                    metadata.last_commit_sha.clone(),
                    i64::from(metadata.file_count),
                    i64::from(metadata.symbol_count),
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
                SELECT file_path, file_hash, indexed_at
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
                indexed_at: parse_timestamp(&row.get::<String>(2).map_err(DatabaseError::from)?)?,
            });
        }

        Ok(files)
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
                    bm25(code_units_fts) AS score
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
                SELECT repo_id, branch, last_indexed_at, last_commit_sha, file_count, symbol_count
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
                last_indexed_at: parse_timestamp(
                    &row.get::<String>(2).map_err(DatabaseError::from)?,
                )?,
                last_commit_sha: row.get::<Option<String>>(3).map_err(DatabaseError::from)?,
                file_count: row.get::<i64>(4).map_err(DatabaseError::from)? as u32,
                symbol_count: row.get::<i64>(5).map_err(DatabaseError::from)? as u32,
            }));
        }

        Ok(None)
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
                    last_indexed_at,
                    last_commit_sha,
                    file_count,
                    symbol_count
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                ON CONFLICT(repo_id, branch) DO UPDATE SET
                    last_indexed_at = excluded.last_indexed_at,
                    last_commit_sha = excluded.last_commit_sha,
                    file_count = excluded.file_count,
                    symbol_count = excluded.symbol_count
                "#,
                params![
                    repo_id,
                    branch,
                    meta.last_indexed_at.to_rfc3339(),
                    meta.last_commit_sha.clone(),
                    i64::from(meta.file_count),
                    i64::from(meta.symbol_count),
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
    ) -> Result<()> {
        Database::upsert_indexed_file(self, repo_id, branch, file_path, file_hash).await
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

async fn run_incremental_migrations(
    connection: &Connection,
) -> std::result::Result<(), DatabaseError> {
    for (version, name, sql) in MIGRATIONS {
        let mut rows = connection
            .query(
                "SELECT 1 FROM _migrations WHERE version = ?1",
                params![version],
            )
            .await?;

        if rows.next().await?.is_some() {
            continue;
        }

        let transaction = connection.transaction().await?;
        transaction.execute_batch(sql).await?;
        transaction
            .execute(
                "INSERT OR IGNORE INTO _migrations (version, name) VALUES (?1, ?2)",
                params![version, name],
            )
            .await?;
        transaction.commit().await?;
    }

    Ok(())
}

fn normalize_repo_root(repo_root: &Path) -> std::result::Result<PathBuf, DatabaseError> {
    if repo_root.exists() {
        return Ok(repo_root.canonicalize()?);
    }

    if repo_root.is_absolute() {
        Ok(repo_root.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(repo_root))
    }
}

fn sanitize_path_segment(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());

    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            sanitized.push(ch);
        } else {
            sanitized.push('-');
        }
    }

    let trimmed = sanitized.trim_matches('-');
    if trimmed.is_empty() {
        String::from("repo")
    } else {
        trimmed.to_owned()
    }
}

fn short_hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();

    let mut hex = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        let _ = write!(&mut hex, "{byte:02x}");
    }

    hex
}

fn long_hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();

    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut hex, "{byte:02x}");
    }

    hex
}

fn code_unit_id(repo_id: &str, branch: &str, unit: &CodeUnit) -> String {
    long_hex_digest(
        format!(
            "{repo_id}\u{1f}{branch}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
            path_to_string(&unit.file_path),
            unit.name,
            code_unit_kind_to_str(unit.kind),
            unit.line_start,
        )
        .as_bytes(),
    )
}

fn default_file_hash(unit: &CodeUnit) -> String {
    long_hex_digest(unit.content.as_bytes())
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn language_to_str(language: Language) -> &'static str {
    match language {
        Language::Rust => "rust",
        Language::TypeScript => "typescript",
        Language::CSharp => "csharp",
    }
}

fn parse_language(value: &str) -> std::result::Result<Language, DatabaseError> {
    match value {
        "rust" => Ok(Language::Rust),
        "typescript" => Ok(Language::TypeScript),
        "csharp" => Ok(Language::CSharp),
        other => Err(DatabaseError::InvalidEmbedding(format!(
            "unknown language value {other}"
        ))),
    }
}

fn code_unit_kind_to_str(kind: CodeUnitKind) -> &'static str {
    match kind {
        CodeUnitKind::Function => "function",
        CodeUnitKind::Method => "method",
        CodeUnitKind::Struct => "struct",
        CodeUnitKind::Enum => "enum",
        CodeUnitKind::Trait => "trait",
        CodeUnitKind::Class => "class",
        CodeUnitKind::Interface => "interface",
        CodeUnitKind::Constant => "constant",
        CodeUnitKind::Module => "module",
    }
}

fn parse_code_unit_kind(value: &str) -> std::result::Result<CodeUnitKind, DatabaseError> {
    match value {
        "function" => Ok(CodeUnitKind::Function),
        "method" => Ok(CodeUnitKind::Method),
        "struct" => Ok(CodeUnitKind::Struct),
        "enum" => Ok(CodeUnitKind::Enum),
        "trait" => Ok(CodeUnitKind::Trait),
        "class" => Ok(CodeUnitKind::Class),
        "interface" => Ok(CodeUnitKind::Interface),
        "constant" => Ok(CodeUnitKind::Constant),
        "module" => Ok(CodeUnitKind::Module),
        other => Err(DatabaseError::InvalidEmbedding(format!(
            "unknown code unit kind value {other}"
        ))),
    }
}

fn code_unit_from_row(row: &libsql::Row, offset: i32) -> Result<CodeUnit> {
    Ok(CodeUnit {
        name: row.get::<String>(offset).map_err(DatabaseError::from)?,
        kind: parse_code_unit_kind(&row.get::<String>(offset + 1).map_err(DatabaseError::from)?)?,
        signature: row.get::<String>(offset + 2).map_err(DatabaseError::from)?,
        doc: row
            .get::<Option<String>>(offset + 3)
            .map_err(DatabaseError::from)?,
        file_path: PathBuf::from(row.get::<String>(offset + 4).map_err(DatabaseError::from)?),
        line_start: row.get::<i64>(offset + 5).map_err(DatabaseError::from)? as u32,
        line_end: row.get::<i64>(offset + 6).map_err(DatabaseError::from)? as u32,
        content: row.get::<String>(offset + 7).map_err(DatabaseError::from)?,
        parent: row
            .get::<Option<String>>(offset + 8)
            .map_err(DatabaseError::from)?,
        language: parse_language(&row.get::<String>(offset + 9).map_err(DatabaseError::from)?)?,
    })
}

fn encode_embedding(embedding: &[f32]) -> std::result::Result<Vec<u8>, DatabaseError> {
    if embedding.is_empty() {
        return Err(DatabaseError::InvalidEmbedding(
            "embedding must not be empty".to_string(),
        ));
    }

    let mut blob = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    Ok(blob)
}

fn decode_embedding(blob: &[u8]) -> Result<Vec<f32>> {
    if !blob.len().is_multiple_of(4) {
        return Err(DatabaseError::InvalidEmbeddingLength(blob.len()).into());
    }

    let mut embedding = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        embedding.push(f32::from_le_bytes(bytes));
    }
    Ok(embedding)
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(DatabaseError::InvalidEmbedding(format!(
            "embedding dimension mismatch: {} != {}",
            left.len(),
            right.len()
        ))
        .into());
    }

    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (left_norm.sqrt() * right_norm.sqrt()))
}

/// Sanitize a user query for safe use in an FTS5 MATCH clause.
///
/// Each whitespace-separated token is wrapped in double quotes so the FTS5
/// tokenizer processes it as a literal phrase. Internal double-quote characters
/// are escaped by doubling them (`""`), which is FTS5's quoting convention.
///
/// Returns `None` when the query contains no usable tokens (empty / whitespace-only).
fn sanitize_fts_query(query: &str) -> Option<String> {
    let tokens: Vec<String> = query
        .split_whitespace()
        .map(|word| {
            let escaped = word.replace('"', "\"\"");
            format!("\"{}\"", escaped)
        })
        .collect();

    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

fn normalize_fts_score(raw_score: f32) -> f32 {
    let magnitude = if raw_score.is_sign_negative() {
        -raw_score
    } else {
        raw_score
    };
    1.0 / (1.0 + magnitude)
}

fn parse_timestamp(value: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|timestamp| timestamp.with_timezone(&Utc))
        .or_else(|_| {
            DateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S%.fZ")
                .map(|timestamp| timestamp.with_timezone(&Utc))
        })
        .map_err(|_| DatabaseError::InvalidTimestamp(value.to_string()).into())
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
        CodeUnit, CodeUnitKind, IndexMetadata, IndexedCodeUnit, Language, MatchType, Storage,
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
        assert_eq!(migrations, vec![(1, String::from("initial_schema"))]);
    }

    #[tokio::test]
    async fn run_migrations_is_idempotent() {
        let (_temp_dir, database) = open_temp_database().await;

        database.run_migrations().await.unwrap();

        let migrations = migration_rows(&database).await;
        assert_eq!(migrations, vec![(1, String::from("initial_schema"))]);

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
        let metadata = IndexMetadata {
            repo_id: REPO_ID.to_string(),
            branch: BRANCH.to_string(),
            last_indexed_at: Utc::now(),
            last_commit_sha: Some("abc123".to_string()),
            file_count: 7,
            symbol_count: 21,
        };

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
            .upsert_indexed_file(REPO_ID, BRANCH, Path::new("src/alpha.rs"), "hash-alpha")
            .await
            .unwrap();
        database
            .upsert_indexed_file(REPO_ID, BRANCH, Path::new("src/beta.rs"), "hash-beta")
            .await
            .unwrap();
        database
            .set_index_metadata(
                REPO_ID,
                BRANCH,
                &IndexMetadata {
                    repo_id: REPO_ID.to_string(),
                    branch: BRANCH.to_string(),
                    last_indexed_at: Utc::now(),
                    last_commit_sha: None,
                    file_count: 2,
                    symbol_count: 3,
                },
            )
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
                .set_index_metadata(
                    REPO_ID,
                    branch,
                    &IndexMetadata {
                        repo_id: REPO_ID.to_string(),
                        branch: branch.to_string(),
                        last_indexed_at: Utc::now(),
                        last_commit_sha: None,
                        file_count: 0,
                        symbol_count: 0,
                    },
                )
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
                &IndexMetadata {
                    repo_id: REPO_ID.to_string(),
                    branch: "main".to_string(),
                    last_indexed_at: Utc::now(),
                    last_commit_sha: Some("1111111".to_string()),
                    file_count: 1,
                    symbol_count: 1,
                },
            )
            .await
            .unwrap();
        database
            .set_index_metadata(
                REPO_ID,
                "feature/auth",
                &IndexMetadata {
                    repo_id: REPO_ID.to_string(),
                    branch: "feature/auth".to_string(),
                    last_indexed_at: Utc::now(),
                    last_commit_sha: Some("2222222".to_string()),
                    file_count: 1,
                    symbol_count: 1,
                },
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
            .upsert_indexed_file(REPO_ID, "main", Path::new("src/lib.rs"), "hash-main")
            .await
            .unwrap();
        database
            .upsert_indexed_file(
                REPO_ID,
                "feature/auth",
                Path::new("src/lib.rs"),
                "hash-feature",
            )
            .await
            .unwrap();
        database
            .set_index_metadata(
                REPO_ID,
                "main",
                &IndexMetadata {
                    repo_id: REPO_ID.to_string(),
                    branch: "main".to_string(),
                    last_indexed_at: Utc::now(),
                    last_commit_sha: None,
                    file_count: 1,
                    symbol_count: 1,
                },
            )
            .await
            .unwrap();
        database
            .set_index_metadata(
                REPO_ID,
                "feature/auth",
                &IndexMetadata {
                    repo_id: REPO_ID.to_string(),
                    branch: "feature/auth".to_string(),
                    last_indexed_at: Utc::now(),
                    last_commit_sha: None,
                    file_count: 1,
                    symbol_count: 1,
                },
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
