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

mod bookkeeping;
mod lifecycle;
mod search;

pub(super) const REPO_ID: &str = "/repo";
pub(super) const BRANCH: &str = "main";

pub(super) async fn open_temp_database() -> (TempDir, Database) {
    let temp_dir = tempfile::tempdir().unwrap();
    let db_path = temp_dir.path().join("storage.db");
    let database = Database::new(&db_path).await.unwrap();
    database.run_migrations().await.unwrap();
    (temp_dir, database)
}

pub(super) async fn sqlite_objects(database: &Database, object_type: &str) -> Vec<String> {
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

pub(super) async fn migration_rows(database: &Database) -> Vec<(i64, String)> {
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

pub(super) fn sample_unit(name: &str, file_path: &str, line_start: u32, content: &str) -> CodeUnit {
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

pub(super) fn sample_metadata(
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

pub(super) async fn stored_embedding_blob(database: &Database, name: &str) -> Vec<u8> {
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
