use std::fs;
use std::path::Path;
use std::sync::OnceLock;

use chrono::{Duration, Utc};
use tokio::sync::Mutex;
use truesight_core::IndexMetadata;

use super::*;

mod git_fixture {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/test_support/git_fixture.rs"
    ));
}

use git_fixture::TempGitFixture;

#[tokio::test]
async fn index_command_indexes_fixture_and_reports_stats() {
    let _guard = test_lock().lock().await;
    let fixture = TempGitFixture::new("rust-fixture");
    cleanup_repo_database(fixture.path());
    let mut output = Vec::new();

    run(
        Cli {
            command: Commands::Index(IndexArgs {
                path: fixture.path_buf(),
                full: true,
            }),
        },
        &mut output,
    )
    .await
    .expect("index command should succeed");

    let text = String::from_utf8(output).expect("index output should be utf-8");
    assert!(text.contains("files_indexed:"));
    assert!(text.contains("symbols_extracted:"));

    let files_indexed =
        parse_numeric_field(&text, "files_indexed").expect("files_indexed should exist");
    let symbols_extracted =
        parse_numeric_field(&text, "symbols_extracted").expect("symbols_extracted should exist");
    assert!(files_indexed > 0);
    assert!(symbols_extracted > 0);
}

#[tokio::test]
async fn search_command_returns_ranked_results_for_indexed_fixture() {
    let _guard = test_lock().lock().await;
    let fixture = TempGitFixture::new("csharp-fixture");
    cleanup_repo_database(fixture.path());
    let mut index_output = Vec::new();

    run(
        Cli {
            command: Commands::Index(IndexArgs {
                path: fixture.path_buf(),
                full: true,
            }),
        },
        &mut index_output,
    )
    .await
    .expect("fixture should index before search");

    let mut search_output = Vec::new();
    run(
        Cli {
            command: Commands::Search(SearchArgs {
                query: "authenticate user".to_string(),
                repo: fixture.path_buf(),
                limit: 5,
            }),
        },
        &mut search_output,
    )
    .await
    .expect("search command should succeed after indexing");

    let text = String::from_utf8(search_output).expect("search output should be utf-8");
    assert!(text.contains("total_results:"));
    assert!(text.contains("AuthService"));
    assert!(text.contains("score="));
}

#[tokio::test]
async fn search_command_suggests_indexing_when_repo_is_missing_metadata() {
    let _guard = test_lock().lock().await;
    let fixture = TempGitFixture::new("ts-fixture");
    cleanup_repo_database(fixture.path());
    let mut output = Vec::new();

    let error = run(
        Cli {
            command: Commands::Search(SearchArgs {
                query: "validate email".to_string(),
                repo: fixture.path_buf(),
                limit: 3,
            }),
        },
        &mut output,
    )
    .await
    .expect_err("search should fail when repo is not indexed");

    let message = error.to_string();
    assert!(message.contains("repository is not indexed"));
    assert!(message.contains("truesight index"));
}

#[tokio::test]
async fn search_command_returns_no_results_for_indexed_empty_repo() {
    let _guard = test_lock().lock().await;
    let temp_dir = tempfile::tempdir().expect("temp dir should exist");
    fs::write(temp_dir.path().join("README.md"), "empty\n").expect("readme write should succeed");
    cleanup_repo_database(temp_dir.path());

    let mut index_output = Vec::new();
    run(
        Cli {
            command: Commands::Index(IndexArgs {
                path: temp_dir.path().to_path_buf(),
                full: true,
            }),
        },
        &mut index_output,
    )
    .await
    .expect("empty repo should still index successfully");

    let mut search_output = Vec::new();
    run(
        Cli {
            command: Commands::Search(SearchArgs {
                query: "anything".to_string(),
                repo: temp_dir.path().to_path_buf(),
                limit: 5,
            }),
        },
        &mut search_output,
    )
    .await
    .expect("empty repo search should return an empty result set");

    let text = String::from_utf8(search_output).expect("search output should be utf-8");
    assert!(text.contains("total_results: 0"));
    assert!(text.contains("no_results: true"));
}

#[tokio::test]
async fn search_command_handles_blank_query_without_crashing() {
    let _guard = test_lock().lock().await;
    let fixture = TempGitFixture::new("rust-fixture");
    cleanup_repo_database(fixture.path());

    let mut index_output = Vec::new();
    run(
        Cli {
            command: Commands::Index(IndexArgs {
                path: fixture.path_buf(),
                full: true,
            }),
        },
        &mut index_output,
    )
    .await
    .expect("fixture should index before empty-query search");

    let mut output = Vec::new();
    run(
        Cli {
            command: Commands::Search(SearchArgs {
                query: String::new(),
                repo: fixture.path_buf(),
                limit: 3,
            }),
        },
        &mut output,
    )
    .await
    .expect("empty query should not crash");

    let text = String::from_utf8(output).expect("search output should be utf-8");
    assert!(text.contains("query: "));
    assert!(text.contains("total_results:"));
}

#[test]
fn stale_detection_requires_age_and_new_commit() {
    let metadata = IndexMetadata {
        repo_id: String::from("repo-id"),
        branch: String::from("main"),
        status: IndexStatus::Ready,
        last_indexed_at: Utc::now() - Duration::minutes(10),
        last_commit_sha: Some(String::from("old")),
        last_seen_commit_sha: Some(String::from("old")),
        file_count: 1,
        symbol_count: 1,
        embedding_model: String::from("all-MiniLM-L6-v2"),
        last_error: None,
    };

    assert!(runtime::should_refresh_index(&metadata, Some("new")));
    assert!(!runtime::should_refresh_index(&metadata, Some("old")));
    assert!(!runtime::should_refresh_index(&metadata, None));
}

#[test]
fn stale_indexing_marker_is_removed() {
    let fixture = TempGitFixture::new("rust-fixture");
    let marker_path =
        runtime::indexing_marker_path(fixture.path()).expect("marker path should resolve");
    if let Some(parent) = marker_path.parent() {
        fs::create_dir_all(parent).expect("marker parent should exist");
    }

    fs::write(&marker_path, b"busy").expect("marker file should be created");
    runtime::clear_stale_indexing_marker(&marker_path, StdDuration::ZERO);

    assert!(
        !marker_path.exists(),
        "stale marker should be removed: {}",
        marker_path.display()
    );
}

#[tokio::test]
async fn open_database_reuses_cached_handle_per_repo_path() {
    let _guard = test_lock().lock().await;
    runtime::clear_runtime_caches().await;

    let fixture = TempGitFixture::new("rust-fixture");
    let first = runtime::open_database(fixture.path())
        .await
        .expect("first open should succeed");
    let second = runtime::open_database(fixture.path())
        .await
        .expect("second open should succeed");

    let _ = (first, second);
    assert_eq!(runtime::cached_database_count().await, 1);
}

#[tokio::test]
async fn repomap_rust_fixture_snapshot() {
    let _guard = test_lock().lock().await;
    let fixture = TempGitFixture::new("rust-fixture");
    cleanup_repo_database(fixture.path());

    let output = run_repomap_for_fixture(&fixture).await;
    let sanitized = sanitize_paths(&output, fixture.path());
    insta::assert_snapshot!("repomap_rust_fixture", sanitized);
}

#[tokio::test]
async fn repomap_csharp_fixture_snapshot() {
    let _guard = test_lock().lock().await;
    let fixture = TempGitFixture::new("csharp-fixture");
    cleanup_repo_database(fixture.path());

    let output = run_repomap_for_fixture(&fixture).await;
    let sanitized = sanitize_paths(&output, fixture.path());
    insta::assert_snapshot!("repomap_csharp_fixture", sanitized);
}

#[tokio::test]
async fn repomap_ts_fixture_snapshot() {
    let _guard = test_lock().lock().await;
    let fixture = TempGitFixture::new("ts-fixture");
    cleanup_repo_database(fixture.path());

    let output = run_repomap_for_fixture(&fixture).await;
    let sanitized = sanitize_paths(&output, fixture.path());
    insta::assert_snapshot!("repomap_ts_fixture", sanitized);
}

async fn run_repomap_for_fixture(fixture: &TempGitFixture) -> String {
    let mut output = Vec::new();
    run(
        Cli {
            command: Commands::RepoMap(RepoMapArgs {
                path: fixture.path_buf(),
                filter: None,
            }),
        },
        &mut output,
    )
    .await
    .expect("repomap command should succeed");
    String::from_utf8(output).expect("repomap output should be utf-8")
}

fn sanitize_paths(output: &str, repo_root: &Path) -> String {
    let canonical = repo_root
        .canonicalize()
        .unwrap_or_else(|_| repo_root.to_path_buf());
    output.replace(canonical.to_str().unwrap(), "<REPO_ROOT>")
}

fn parse_numeric_field(output: &str, name: &str) -> Option<u64> {
    output.lines().find_map(|line| {
        let (field, value) = line.split_once(':')?;
        (field.trim() == name)
            .then(|| value.trim().parse().ok())
            .flatten()
    })
}

fn cleanup_repo_database(repo_root: &Path) {
    if let Ok(db_path) = Database::db_path_for_repo(repo_root) {
        let _ = fs::remove_file(&db_path);
        let _ = fs::remove_file(format!("{}-wal", db_path.display()));
        let _ = fs::remove_file(format!("{}-shm", db_path.display()));
    }
}

fn test_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::const_new(()))
}
