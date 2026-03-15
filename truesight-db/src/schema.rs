use super::*;

pub(super) const MIGRATION_TABLE_SQL: &str = r#"
CREATE TABLE IF NOT EXISTS _migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);
"#;

pub(super) const SCHEMA: &str = r#"
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

pub(super) const MIGRATION_ADD_INDEX_STATE_SQL: &str = r#"
ALTER TABLE index_metadata ADD COLUMN status TEXT NOT NULL DEFAULT 'ready';
ALTER TABLE index_metadata ADD COLUMN last_seen_commit_sha TEXT;
ALTER TABLE index_metadata ADD COLUMN embedding_model TEXT NOT NULL DEFAULT '';
ALTER TABLE index_metadata ADD COLUMN last_error TEXT;
ALTER TABLE code_units ADD COLUMN embedding_model TEXT NOT NULL DEFAULT '';
ALTER TABLE indexed_files ADD COLUMN chunk_count INTEGER NOT NULL DEFAULT 0;
"#;

pub(super) const MIGRATIONS: &[(i64, &str, &str)] = &[
    (1, "initial_schema", SCHEMA),
    (
        2,
        "add_index_state_and_embedding_metadata",
        MIGRATION_ADD_INDEX_STATE_SQL,
    ),
];

pub(super) async fn run_incremental_migrations(
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

pub(super) async fn ensure_vector_index(
    connection: &Connection,
) -> std::result::Result<(), DatabaseError> {
    match connection
        .execute(
            "CREATE INDEX IF NOT EXISTS idx_code_units_embedding ON code_units(libsql_vector_idx(embedding))",
            (),
        )
        .await
    {
        Ok(_) => Ok(()),
        Err(error) => {
            warn!(error = %error, "vector index unavailable; semantic search will use fallback path");
            Ok(())
        }
    }
}
