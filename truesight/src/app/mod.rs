use std::collections::HashMap;
use std::future::Future;
use std::io::Write;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration as StdDuration;

use anyhow::{Context, anyhow};
use serde::Serialize;
use truesight_core::{Embedder, IndexStatus};
use truesight_core::{IndexStats, RepoMap, SearchConfig, SearchResult, Storage};
use truesight_db::Database;
use truesight_engine::{
    IncrementalIndexer, OnnxEmbedder, RepoContext, RepoMapGenerator, SearchEngine,
    detect_repo_context_from_root, index_repo,
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
        let ((context, database), embedder) =
            tokio::try_join!(prepare_context_and_database(&repo_root), load_embedder(),)?;

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
        let embedder = load_embedder().await?;
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
        retry_repo_operation(
            &repo_root,
            format!("timed out preparing index for {}", repo_root.display()),
            |repo_root, context, database| {
                Box::pin(async move {
                    let metadata =
                        Storage::get_index_metadata(&database, &context.repo_id, &context.branch)
                            .await?;

                    let should_reindex = match metadata.as_ref() {
                        Some(metadata) => {
                            should_refresh_index(metadata, context.last_commit_sha.as_deref())
                        }
                        None => {
                            !branch_has_index(&database, &context.repo_id, &context.branch).await?
                        }
                    };

                    if should_reindex {
                        let _ = self
                            .index_repo_response_from_root(repo_root.clone(), false)
                            .await?;
                    }

                    Ok::<_, anyhow::Error>((repo_root.clone(), context, database))
                })
            },
        )
        .await
    }

    async fn search_repo_for_cli_from_root(
        &self,
        repo_root: PathBuf,
        query: String,
        limit: usize,
    ) -> anyhow::Result<SearchRepoResponse> {
        retry_repo_operation(
            &repo_root,
            format!(
                "timed out searching while index was being prepared for {}",
                repo_root.display()
            ),
            |repo_root, context, database| {
                let query = query.clone();
                Box::pin(async move {
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
                })
            },
        )
        .await
    }
}

type RetryFuture<'a, T> = Pin<Box<dyn Future<Output = anyhow::Result<T>> + Send + 'a>>;

async fn retry_repo_operation<'a, T, F>(
    repo_root: &'a PathBuf,
    timeout_message: String,
    mut operation: F,
) -> anyhow::Result<T>
where
    F: FnMut(PathBuf, RepoContext, Database) -> RetryFuture<'a, T>,
{
    for attempt in 0..=SEARCH_RETRY_ATTEMPTS {
        wait_for_indexing_marker(repo_root).await?;

        let (context, database) = prepare_context_and_database(repo_root).await?;
        let result = operation(repo_root.clone(), context, database).await;

        match result {
            Ok(value) => return Ok(value),
            Err(error)
                if should_retry_repo_operation(&error) && attempt < SEARCH_RETRY_ATTEMPTS =>
            {
                tokio::time::sleep(StdDuration::from_millis(INDEXING_WAIT_INTERVAL_MS)).await;
            }
            Err(error) => return Err(error),
        }
    }

    Err(anyhow!(timeout_message))
}

async fn prepare_context_and_database(
    repo_root: &PathBuf,
) -> anyhow::Result<(RepoContext, Database)> {
    let repo_root_for_context = repo_root.clone();
    let context_task =
        tokio::task::spawn_blocking(move || detect_repo_context_from_root(&repo_root_for_context));

    let database = open_database(repo_root).await?;
    let context = context_task
        .await
        .map_err(|error| anyhow!("repo context task failed: {error}"))??;

    Ok((context, database))
}

async fn load_embedder() -> anyhow::Result<OnnxEmbedder> {
    tokio::task::spawn_blocking(OnnxEmbedder::new)
        .await
        .map_err(|error| anyhow!("embedder task failed: {error}"))?
        .context("failed to initialize embedder")
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
mod tests;
