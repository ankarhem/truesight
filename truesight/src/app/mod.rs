use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration as StdDuration;

use anyhow::{Context, anyhow};
use serde::Serialize;
use truesight_core::{Embedder, IndexStatus};
use truesight_core::{IndexStats, RepoMap, SearchConfig, SearchResult, Storage};
use truesight_db::Database;
use truesight_engine::{
    IncrementalIndexer, OnnxEmbedder, RepoContext, RepoMapGenerator, SearchEngine,
    detect_repo_context, index_repo,
};

use crate::cli::{Cli, Commands, IndexArgs, RepoMapArgs, SearchArgs};
mod output;
mod runtime;

use output::{write_index_output, write_repomap_output, write_search_output};
use runtime::{
    INDEXING_WAIT_INTERVAL_MS, branch_has_index, canonicalize_repo_root, change_count,
    lock_repo_operation, open_database, should_refresh_index, should_retry_repo_operation,
    wait_for_indexing_marker,
};

const SEARCH_RETRY_ATTEMPTS: usize = 10;

#[derive(Debug, Clone, Default)]
pub(crate) struct AppServices;

impl AppServices {
    pub(crate) fn new() -> Self {
        Self
    }

    pub(crate) async fn index_repo_response(
        &self,
        path: PathBuf,
        full: bool,
    ) -> anyhow::Result<IndexRepoResponse> {
        let repo_root = canonicalize_repo_root(&path)?;
        let _guard = lock_repo_operation(&repo_root).await;
        self.index_repo_response_from_root(repo_root, full).await
    }

    async fn index_repo_response_from_root(
        &self,
        repo_root: PathBuf,
        full: bool,
    ) -> anyhow::Result<IndexRepoResponse> {
        let _marker = runtime::acquire_indexing_marker(&repo_root).await?;
        let context = detect_repo_context(&repo_root)?;
        let database = open_database(&repo_root).await?;
        let embedder = OnnxEmbedder::new().context("failed to initialize embedder")?;

        database
            .update_index_status(
                &context.repo_id,
                &context.branch,
                IndexStatus::Indexing,
                context.last_commit_sha.as_deref(),
                Some(embedder.model_name()),
                None,
            )
            .await?;

        let stats_result = async {
            if full {
                index_repo(&repo_root, &database, &embedder).await
            } else {
                let indexer = IncrementalIndexer::new();
                let changes = indexer
                    .detect_changes(&repo_root, &database, &context.repo_id, &context.branch)
                    .await?;

                if change_count(&changes) == 0 {
                    Ok(IndexStats {
                        files_scanned: 0,
                        files_indexed: 0,
                        files_skipped: 0,
                        symbols_extracted: 0,
                        chunks_embedded: 0,
                        duration_ms: 0,
                        languages: HashMap::new(),
                    })
                } else {
                    indexer
                        .incremental_index(
                            &repo_root,
                            &changes,
                            &database,
                            &embedder,
                            &context.repo_id,
                            &context.branch,
                        )
                        .await
                }
            }
        }
        .await;

        let stats = match stats_result {
            Ok(stats) => stats,
            Err(error) => {
                let _ = database
                    .update_index_status(
                        &context.repo_id,
                        &context.branch,
                        IndexStatus::Failed,
                        context.last_commit_sha.as_deref(),
                        Some(embedder.model_name()),
                        Some(&error.to_string()),
                    )
                    .await;
                return Err(error.into());
            }
        };

        Ok(IndexRepoResponse::from_stats(
            repo_root,
            context.branch,
            stats,
        ))
    }

    pub(crate) async fn search_repo_for_cli(
        &self,
        path: PathBuf,
        query: String,
        limit: usize,
    ) -> anyhow::Result<SearchRepoResponse> {
        let repo_root = canonicalize_repo_root(&path)?;
        let _guard = lock_repo_operation(&repo_root).await;
        self.search_repo_for_cli_from_root(repo_root, query, limit)
            .await
    }

    pub(crate) async fn search_repo_response(
        &self,
        path: PathBuf,
        query: String,
        limit: usize,
    ) -> anyhow::Result<SearchRepoResponse> {
        let repo_root = canonicalize_repo_root(&path)?;
        let _guard = lock_repo_operation(&repo_root).await;
        let (repo_root, context, database) = self.ensure_index_ready_from_root(repo_root).await?;
        self.execute_search(repo_root, context, database, query, limit)
            .await
    }

    pub(crate) async fn repo_map_response(
        &self,
        path: PathBuf,
        filter: Option<String>,
    ) -> anyhow::Result<RepoMap> {
        let repo_root = canonicalize_repo_root(&path)?;
        let _guard = lock_repo_operation(&repo_root).await;
        let (repo_root, context, database) = self.ensure_index_ready_from_root(repo_root).await?;
        RepoMapGenerator::generate(
            &repo_root,
            &database,
            &context.repo_id,
            &context.branch,
            filter.as_deref(),
        )
        .await
        .map_err(Into::into)
    }

    async fn execute_search(
        &self,
        repo_root: PathBuf,
        context: RepoContext,
        database: Database,
        query: String,
        limit: usize,
    ) -> anyhow::Result<SearchRepoResponse> {
        let embedder = OnnxEmbedder::new().context("failed to initialize embedder")?;
        let search_engine = SearchEngine::new(&database, Some(&embedder));
        let config = SearchConfig {
            limit,
            ..SearchConfig::default()
        };
        let results = search_engine
            .search(&query, &context.repo_id, &context.branch, &config)
            .await?;
        let total_results = results.len();

        Ok(SearchRepoResponse {
            query,
            results,
            total_results,
            search_mode: String::from("hybrid"),
            repo_root,
            branch: context.branch,
        })
    }

    async fn ensure_index_ready_from_root(
        &self,
        repo_root: PathBuf,
    ) -> anyhow::Result<(PathBuf, RepoContext, Database)> {
        for attempt in 0..=SEARCH_RETRY_ATTEMPTS {
            wait_for_indexing_marker(&repo_root).await?;

            let context = detect_repo_context(&repo_root)?;
            let database = open_database(&repo_root).await?;
            let result = async {
                let metadata =
                    Storage::get_index_metadata(&database, &context.repo_id, &context.branch)
                        .await?;

                let should_reindex = match metadata.as_ref() {
                    Some(metadata) => {
                        should_refresh_index(metadata, context.last_commit_sha.as_deref())
                    }
                    None => !branch_has_index(&database, &context.repo_id, &context.branch).await?,
                };

                if should_reindex {
                    let _ = self
                        .index_repo_response_from_root(repo_root.clone(), false)
                        .await?;
                }

                Ok::<_, anyhow::Error>((repo_root.clone(), context, database))
            }
            .await;

            match result {
                Ok(ready) => return Ok(ready),
                Err(error)
                    if should_retry_repo_operation(&error) && attempt < SEARCH_RETRY_ATTEMPTS =>
                {
                    tokio::time::sleep(StdDuration::from_millis(INDEXING_WAIT_INTERVAL_MS)).await;
                }
                Err(error) => return Err(error),
            }
        }

        Err(anyhow!(
            "timed out preparing index for {}",
            repo_root.display()
        ))
    }

    async fn search_repo_for_cli_from_root(
        &self,
        repo_root: PathBuf,
        query: String,
        limit: usize,
    ) -> anyhow::Result<SearchRepoResponse> {
        for attempt in 0..=SEARCH_RETRY_ATTEMPTS {
            wait_for_indexing_marker(&repo_root).await?;

            let context = detect_repo_context(&repo_root)?;
            let database = open_database(&repo_root).await?;
            let result = async {
                let metadata =
                    Storage::get_index_metadata(&database, &context.repo_id, &context.branch)
                        .await?;

                if metadata.is_none()
                    && !branch_has_index(&database, &context.repo_id, &context.branch).await?
                {
                    return Err(anyhow!(
                        "repository is not indexed for branch `{}`; run `truesight index {}` first",
                        context.branch,
                        repo_root.display()
                    ));
                }

                self.execute_search(repo_root.clone(), context, database, query.clone(), limit)
                    .await
            }
            .await;

            match result {
                Ok(response) => return Ok(response),
                Err(error)
                    if should_retry_repo_operation(&error) && attempt < SEARCH_RETRY_ATTEMPTS =>
                {
                    tokio::time::sleep(StdDuration::from_millis(INDEXING_WAIT_INTERVAL_MS)).await;
                }
                Err(error) => return Err(error),
            }
        }

        Err(anyhow!(
            "timed out searching while index was being prepared for {}",
            repo_root.display()
        ))
    }
}

pub(crate) async fn run(cli: Cli, writer: &mut dyn Write) -> anyhow::Result<()> {
    let app = AppServices::new();

    match cli.command {
        Commands::Mcp => crate::mcp::run().await,
        Commands::Index(args) => run_index(&app, args, writer).await,
        Commands::Search(args) => run_search(&app, args, writer).await,
        Commands::RepoMap(args) => run_repomap(&app, args, writer).await,
    }
}

async fn run_index(
    app: &AppServices,
    args: IndexArgs,
    writer: &mut dyn Write,
) -> anyhow::Result<()> {
    let response = app.index_repo_response(args.path, args.full).await?;
    write_index_output(writer, &response, args.full)
}

async fn run_search(
    app: &AppServices,
    args: SearchArgs,
    writer: &mut dyn Write,
) -> anyhow::Result<()> {
    let response = app
        .search_repo_for_cli(args.repo, args.query, args.limit)
        .await?;
    write_search_output(writer, &response)
}

async fn run_repomap(
    app: &AppServices,
    args: RepoMapArgs,
    writer: &mut dyn Write,
) -> anyhow::Result<()> {
    let response = app.repo_map_response(args.path, args.filter).await?;
    write_repomap_output(writer, &response)
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SearchRepoResponse {
    pub(crate) query: String,
    pub(crate) results: Vec<SearchResult>,
    pub(crate) total_results: usize,
    pub(crate) search_mode: String,
    #[serde(skip_serializing)]
    pub(crate) repo_root: PathBuf,
    #[serde(skip_serializing)]
    pub(crate) branch: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct IndexRepoResponse {
    pub(crate) status: String,
    pub(crate) repo_root: PathBuf,
    pub(crate) branch: String,
    pub(crate) stats: IndexRepoStats,
    pub(crate) languages: HashMap<String, u32>,
}

impl IndexRepoResponse {
    fn from_stats(repo_root: PathBuf, branch: String, stats: IndexStats) -> Self {
        let mut languages = HashMap::new();
        for (language, count) in stats.languages {
            languages.insert(language.to_string(), count);
        }

        Self {
            status: String::from("completed"),
            repo_root,
            branch,
            stats: IndexRepoStats {
                files_scanned: stats.files_scanned,
                files_indexed: stats.files_indexed,
                files_skipped: stats.files_skipped,
                symbols_extracted: stats.symbols_extracted,
                chunks_embedded: stats.chunks_embedded,
                duration_ms: stats.duration_ms,
            },
            languages,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct IndexRepoStats {
    pub(crate) files_scanned: u32,
    pub(crate) files_indexed: u32,
    pub(crate) files_skipped: u32,
    pub(crate) symbols_extracted: u32,
    pub(crate) chunks_embedded: u32,
    pub(crate) duration_ms: u64,
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::OnceLock;

    use chrono::{Duration, Utc};
    use truesight_core::IndexMetadata;

    mod git_fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/test_support/git_fixture.rs"
        ));
    }

    use tokio::sync::Mutex;

    use super::*;
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
        let symbols_extracted = parse_numeric_field(&text, "symbols_extracted")
            .expect("symbols_extracted should exist");
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
        fs::write(temp_dir.path().join("README.md"), "empty\n")
            .expect("readme write should succeed");
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
            // Also remove SQLite WAL and SHM files to ensure clean state
            let _ = fs::remove_file(format!("{}-wal", db_path.display()));
            let _ = fs::remove_file(format!("{}-shm", db_path.display()));
        }
    }

    fn test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::const_new(()))
    }
}
