use super::*;

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
