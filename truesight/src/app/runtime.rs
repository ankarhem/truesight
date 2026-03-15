use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration as StdDuration;

use anyhow::{Context, Result, anyhow};
use chrono::{Duration, Utc};
use truesight_core::{IndexMetadata, IndexStatus, Storage};
use truesight_db::Database;

const STALE_INDEX_MAX_AGE_MINUTES: i64 = 5;
const INDEXING_WAIT_RETRIES: usize = 600;
pub(super) const INDEXING_WAIT_INTERVAL_MS: u64 = 100;
const INDEXING_MARKER_STALE_SECS: u64 = 300;
static REPO_OPERATION_LOCKS: OnceLock<
    tokio::sync::Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>,
> = OnceLock::new();
static OPEN_DATABASES: OnceLock<tokio::sync::Mutex<HashMap<PathBuf, Database>>> = OnceLock::new();
static MIGRATED_DATABASES: OnceLock<tokio::sync::Mutex<HashSet<PathBuf>>> = OnceLock::new();

pub(super) async fn open_database(repo_root: &Path) -> Result<Database> {
    let db_path = Database::db_path_for_repo(repo_root)?;
    if let Some(database) = cached_database(&db_path).await {
        return Ok(database);
    }

    let database = Database::new(&db_path).await?;

    if migrations_cached(&db_path).await && has_migration_table(&database).await? {
        cache_database(db_path.clone(), &database).await;
        return Ok(database);
    }

    let mut attempts = 0_u8;
    loop {
        match database.run_migrations().await {
            Ok(()) => {
                mark_migrations_cached(db_path.clone()).await;
                cache_database(db_path.clone(), &database).await;
                return Ok(database);
            }
            Err(error) if error.to_string().contains("database is locked") && attempts < 20 => {
                attempts += 1;
                tokio::time::sleep(StdDuration::from_millis(100)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

async fn cached_database(db_path: &Path) -> Option<Database> {
    let mut databases = OPEN_DATABASES
        .get_or_init(|| tokio::sync::Mutex::new(HashMap::new()))
        .lock()
        .await;

    if db_path != Path::new(":memory:") && !db_path.exists() {
        databases.remove(db_path);
        remove_cached_migrations(db_path).await;
        return None;
    }

    databases.get(db_path).cloned()
}

async fn cache_database(db_path: PathBuf, database: &Database) {
    OPEN_DATABASES
        .get_or_init(|| tokio::sync::Mutex::new(HashMap::new()))
        .lock()
        .await
        .insert(db_path, database.clone());
}

async fn migrations_cached(db_path: &Path) -> bool {
    MIGRATED_DATABASES
        .get_or_init(|| tokio::sync::Mutex::new(HashSet::new()))
        .lock()
        .await
        .contains(db_path)
}

async fn mark_migrations_cached(db_path: PathBuf) {
    MIGRATED_DATABASES
        .get_or_init(|| tokio::sync::Mutex::new(HashSet::new()))
        .lock()
        .await
        .insert(db_path);
}

async fn remove_cached_migrations(db_path: &Path) {
    MIGRATED_DATABASES
        .get_or_init(|| tokio::sync::Mutex::new(HashSet::new()))
        .lock()
        .await
        .remove(db_path);
}

async fn has_migration_table(database: &Database) -> Result<bool> {
    let connection = database.connect().await?;
    let mut rows = connection
        .query(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = '_migrations' LIMIT 1",
            (),
        )
        .await
        .map_err(|error| anyhow!(error.to_string()))?;
    Ok(rows
        .next()
        .await
        .map_err(|error| anyhow!(error.to_string()))?
        .is_some())
}

#[cfg(test)]
pub(super) async fn clear_runtime_caches() {
    OPEN_DATABASES
        .get_or_init(|| tokio::sync::Mutex::new(HashMap::new()))
        .lock()
        .await
        .clear();
    MIGRATED_DATABASES
        .get_or_init(|| tokio::sync::Mutex::new(HashSet::new()))
        .lock()
        .await
        .clear();
}

#[cfg(test)]
pub(super) async fn cached_database_count() -> usize {
    OPEN_DATABASES
        .get_or_init(|| tokio::sync::Mutex::new(HashMap::new()))
        .lock()
        .await
        .len()
}

pub(super) async fn branch_has_index(
    database: &Database,
    repo_id: &str,
    branch: &str,
) -> Result<bool> {
    Ok(Storage::has_indexed_symbols(database, repo_id, branch).await?)
}

pub(super) async fn lock_repo_operation(repo_root: &Path) -> tokio::sync::OwnedMutexGuard<()> {
    let map = REPO_OPERATION_LOCKS.get_or_init(|| tokio::sync::Mutex::new(HashMap::new()));
    let repo_lock = {
        let mut locks = map.lock().await;
        locks
            .entry(repo_root.to_path_buf())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    };

    repo_lock.lock_owned().await
}

pub(super) struct IndexingMarker {
    path: PathBuf,
}

impl Drop for IndexingMarker {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

pub(super) async fn acquire_indexing_marker(repo_root: &Path) -> Result<IndexingMarker> {
    let marker_path = indexing_marker_path(repo_root)?;

    for _ in 0..INDEXING_WAIT_RETRIES {
        clear_stale_indexing_marker(
            &marker_path,
            StdDuration::from_secs(INDEXING_MARKER_STALE_SECS),
        );

        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker_path)
        {
            Ok(_) => return Ok(IndexingMarker { path: marker_path }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                tokio::time::sleep(StdDuration::from_millis(INDEXING_WAIT_INTERVAL_MS)).await;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to create index marker {}", marker_path.display())
                });
            }
        }
    }

    Err(anyhow!(
        "timed out waiting for active index to finish for {}",
        repo_root.display()
    ))
}

pub(super) async fn wait_for_indexing_marker(repo_root: &Path) -> Result<()> {
    let marker_path = indexing_marker_path(repo_root)?;

    for _ in 0..INDEXING_WAIT_RETRIES {
        clear_stale_indexing_marker(
            &marker_path,
            StdDuration::from_secs(INDEXING_MARKER_STALE_SECS),
        );

        if !marker_path.exists() {
            return Ok(());
        }

        tokio::time::sleep(StdDuration::from_millis(INDEXING_WAIT_INTERVAL_MS)).await;
    }

    Err(anyhow!(
        "timed out waiting for active index to finish for {}",
        repo_root.display()
    ))
}

pub(super) fn indexing_marker_path(repo_root: &Path) -> Result<PathBuf> {
    let db_path = Database::db_path_for_repo(repo_root)?;
    Ok(PathBuf::from(format!("{}.indexing", db_path.display())))
}

pub(super) fn should_retry_repo_operation(error: &anyhow::Error) -> bool {
    let message = error.to_string();
    message.contains("database is locked")
        || message.contains("no such table")
        || message.contains("repository is not indexed for branch")
}

pub(super) fn clear_stale_indexing_marker(marker_path: &Path, stale_after: StdDuration) {
    let is_stale = fs::metadata(marker_path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|elapsed| elapsed >= stale_after);

    if is_stale {
        let _ = fs::remove_file(marker_path);
    }
}

pub(super) fn canonicalize_repo_root(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("failed to resolve repository path {}", path.display()))
}

pub(super) fn change_count(changes: &truesight_engine::ChangeSet) -> usize {
    changes.added.len() + changes.modified.len() + changes.deleted.len()
}

pub(super) fn should_refresh_index(
    metadata: &IndexMetadata,
    current_commit_sha: Option<&str>,
) -> bool {
    if metadata.status != IndexStatus::Ready {
        return true;
    }

    if Utc::now() - metadata.last_indexed_at < Duration::minutes(STALE_INDEX_MAX_AGE_MINUTES) {
        return false;
    }

    matches!(
        (metadata.last_commit_sha.as_deref(), current_commit_sha),
        (Some(indexed_sha), Some(current_sha)) if indexed_sha != current_sha
    )
}
