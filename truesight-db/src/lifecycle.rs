use super::*;

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
}
