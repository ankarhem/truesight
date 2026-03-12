use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration as StdDuration;

use anyhow::{Context, anyhow};
use chrono::{Duration, Utc};
use serde::Serialize;
use truesight_core::IndexMetadata;
use truesight_core::{
    IndexStats, Language, MatchType, RepoMap, SearchConfig, SearchResult, Storage,
};
use truesight_db::Database;
use truesight_engine::{
    IncrementalIndexer, OnnxEmbedder, RepoContext, RepoMapGenerator, SearchEngine,
    detect_repo_context, index_repo,
};

use crate::cli::{Cli, Commands, IndexArgs, RepoMapArgs, SearchArgs};

const STALE_INDEX_MAX_AGE_MINUTES: i64 = 5;
const INDEXING_WAIT_RETRIES: usize = 600;
const INDEXING_WAIT_INTERVAL_MS: u64 = 100;
const INDEXING_MARKER_STALE_SECS: u64 = 300;
const SEARCH_RETRY_ATTEMPTS: usize = 10;
static REPO_OPERATION_LOCKS: OnceLock<
    tokio::sync::Mutex<HashMap<PathBuf, Arc<tokio::sync::Mutex<()>>>>,
> = OnceLock::new();

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
        let _marker = acquire_indexing_marker(&repo_root).await?;
        let context = detect_repo_context(&repo_root)?;
        let database = open_database(&repo_root).await?;
        let embedder = OnnxEmbedder::new().context("failed to initialize embedder")?;

        let stats = if full {
            index_repo(&repo_root, &database, &embedder).await?
        } else {
            let indexer = IncrementalIndexer::new();
            let changes = indexer
                .detect_changes(&repo_root, &database, &context.repo_id, &context.branch)
                .await?;

            if change_count(&changes) == 0 {
                IndexStats {
                    files_scanned: 0,
                    files_indexed: 0,
                    files_skipped: 0,
                    symbols_extracted: 0,
                    chunks_embedded: 0,
                    duration_ms: 0,
                    languages: HashMap::new(),
                }
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
                    .await?
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

    pub(crate) async fn repo_map_response(&self, path: PathBuf) -> anyhow::Result<RepoMap> {
        let repo_root = canonicalize_repo_root(&path)?;
        let _guard = lock_repo_operation(&repo_root).await;
        let (repo_root, context, database) = self.ensure_index_ready_from_root(repo_root).await?;
        RepoMapGenerator::generate(&repo_root, &database, &context.repo_id, &context.branch)
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
    let response = app.repo_map_response(args.path).await?;
    write_repomap_output(writer, &response)
}

async fn open_database(repo_root: &Path) -> anyhow::Result<Database> {
    let db_path = Database::db_path_for_repo(repo_root)?;
    let database = Database::new(&db_path).await?;

    let mut attempts = 0_u8;
    loop {
        match database.run_migrations().await {
            Ok(()) => return Ok(database),
            Err(error) if error.to_string().contains("database is locked") && attempts < 20 => {
                attempts += 1;
                tokio::time::sleep(StdDuration::from_millis(100)).await;
            }
            Err(error) => return Err(error.into()),
        }
    }
}

async fn branch_has_index(
    database: &Database,
    repo_id: &str,
    branch: &str,
) -> anyhow::Result<bool> {
    Ok(!Storage::get_all_symbols(database, repo_id, branch)
        .await?
        .is_empty())
}

fn write_index_output(
    writer: &mut dyn Write,
    response: &IndexRepoResponse,
    full: bool,
) -> anyhow::Result<()> {
    writeln!(writer, "repo_root: {}", response.repo_root.display())?;
    writeln!(writer, "branch: {}", response.branch)?;
    writeln!(
        writer,
        "mode: {}",
        if full { "full" } else { "incremental" }
    )?;
    writeln!(writer, "files_scanned: {}", response.stats.files_scanned)?;
    writeln!(writer, "files_indexed: {}", response.stats.files_indexed)?;
    writeln!(writer, "files_skipped: {}", response.stats.files_skipped)?;
    writeln!(
        writer,
        "symbols_extracted: {}",
        response.stats.symbols_extracted
    )?;
    writeln!(
        writer,
        "chunks_embedded: {}",
        response.stats.chunks_embedded
    )?;
    writeln!(writer, "duration_ms: {}", response.stats.duration_ms)?;
    writeln!(writer, "languages:")?;

    let mut items = response.languages.iter().collect::<Vec<_>>();
    items.sort_by(|left, right| left.0.cmp(right.0));
    for (language, count) in items {
        writeln!(writer, "- {language}: {count}")?;
    }

    Ok(())
}

async fn lock_repo_operation(repo_root: &Path) -> tokio::sync::OwnedMutexGuard<()> {
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

struct IndexingMarker {
    path: PathBuf,
}

impl Drop for IndexingMarker {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

async fn acquire_indexing_marker(repo_root: &Path) -> anyhow::Result<IndexingMarker> {
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

async fn wait_for_indexing_marker(repo_root: &Path) -> anyhow::Result<()> {
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

fn indexing_marker_path(repo_root: &Path) -> anyhow::Result<PathBuf> {
    let db_path = Database::db_path_for_repo(repo_root)?;
    Ok(PathBuf::from(format!("{}.indexing", db_path.display())))
}

fn should_retry_repo_operation(error: &anyhow::Error) -> bool {
    let message = error.to_string();
    message.contains("database is locked")
        || message.contains("no such table")
        || message.contains("repository is not indexed for branch")
}

fn clear_stale_indexing_marker(marker_path: &Path, stale_after: StdDuration) {
    let is_stale = fs::metadata(marker_path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.elapsed().ok())
        .is_some_and(|elapsed| elapsed >= stale_after);

    if is_stale {
        let _ = fs::remove_file(marker_path);
    }
}

fn write_search_output(
    writer: &mut dyn Write,
    response: &SearchRepoResponse,
) -> anyhow::Result<()> {
    writeln!(writer, "query: {}", response.query)?;
    writeln!(writer, "repo_root: {}", response.repo_root.display())?;
    writeln!(writer, "branch: {}", response.branch)?;
    writeln!(writer, "total_results: {}", response.total_results)?;

    if response.results.is_empty() {
        writeln!(writer, "no_results: true")?;
        return Ok(());
    }

    for result in &response.results {
        let display_path = display_path(&response.repo_root, &result.path);
        writeln!(
            writer,
            "- {} [{}] {}:{} score={:.3} match={}",
            result.name,
            kind_label(result),
            display_path,
            result.line,
            result.score,
            match_type_label(result.match_type)
        )?;
        writeln!(writer, "  signature: {}", result.signature.trim())?;

        if let Some(doc) = result.doc.as_deref().filter(|doc| !doc.trim().is_empty()) {
            writeln!(writer, "  doc: {}", single_line(doc))?;
        }

        writeln!(writer, "  snippet: {}", single_line(&result.snippet))?;
    }

    Ok(())
}

fn write_repomap_output(writer: &mut dyn Write, response: &RepoMap) -> anyhow::Result<()> {
    writeln!(writer, "repo_root: {}", response.repo_root.display())?;
    writeln!(writer, "branch: {}", response.branch)?;
    writeln!(writer, "modules: {}", response.modules.len())?;

    for module in &response.modules {
        let module_path = display_path(&response.repo_root, &module.path);
        writeln!(writer, "\n## {}", module.name)?;
        writeln!(writer, "path: {module_path}")?;

        if !module.files.is_empty() {
            writeln!(writer, "files:")?;
            for file in &module.files {
                writeln!(writer, "  - {file}")?;
            }
        }

        if !module.symbols.is_empty() {
            writeln!(writer, "symbols:")?;
            for sym in &module.symbols {
                writeln!(
                    writer,
                    "  - {} [{}] {}:{} {}",
                    sym.name,
                    kind_label_raw(sym.kind),
                    sym.file,
                    sym.line,
                    sym.signature.trim()
                )?;
            }
        }

        if !module.depends_on.is_empty() {
            writeln!(writer, "depends_on: {}", module.depends_on.join(", "))?;
        }
    }

    Ok(())
}

fn display_path(repo_root: &Path, path: &Path) -> String {
    if path.is_absolute() {
        path.strip_prefix(repo_root)
            .map(|relative| relative.display().to_string())
            .unwrap_or_else(|_| path.display().to_string())
    } else {
        path.display().to_string()
    }
}

fn single_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn kind_label(result: &SearchResult) -> &'static str {
    kind_label_raw(result.kind)
}

fn kind_label_raw(kind: truesight_core::CodeUnitKind) -> &'static str {
    match kind {
        truesight_core::CodeUnitKind::Function => "function",
        truesight_core::CodeUnitKind::Method => "method",
        truesight_core::CodeUnitKind::Struct => "struct",
        truesight_core::CodeUnitKind::Enum => "enum",
        truesight_core::CodeUnitKind::Trait => "trait",
        truesight_core::CodeUnitKind::Class => "class",
        truesight_core::CodeUnitKind::Interface => "interface",
        truesight_core::CodeUnitKind::Constant => "constant",
        truesight_core::CodeUnitKind::Module => "module",
    }
}

fn match_type_label(match_type: MatchType) -> &'static str {
    match match_type {
        MatchType::Fts => "fts",
        MatchType::Vector => "vector",
        MatchType::Hybrid => "hybrid",
    }
}

fn canonicalize_repo_root(path: &Path) -> anyhow::Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("failed to resolve repository path {}", path.display()))
}

fn change_count(changes: &truesight_engine::ChangeSet) -> usize {
    changes.added.len() + changes.modified.len() + changes.deleted.len()
}

fn should_refresh_index(metadata: &IndexMetadata, current_commit_sha: Option<&str>) -> bool {
    if Utc::now() - metadata.last_indexed_at < Duration::minutes(STALE_INDEX_MAX_AGE_MINUTES) {
        return false;
    }

    matches!(
        (metadata.last_commit_sha.as_deref(), current_commit_sha),
        (Some(indexed_sha), Some(current_sha)) if indexed_sha != current_sha
    )
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
            languages.insert(language_label(language).to_string(), count);
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

fn language_label(language: Language) -> &'static str {
    match language {
        Language::CSharp => "csharp",
        Language::Rust => "rust",
        Language::TypeScript => "typescript",
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::OnceLock;

    mod git_fixture {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../tests/test_support/git_fixture.rs"
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
    }

    #[test]
    fn stale_detection_requires_age_and_new_commit() {
        let metadata = IndexMetadata {
            repo_id: String::from("repo-id"),
            branch: String::from("main"),
            last_indexed_at: Utc::now() - Duration::minutes(10),
            last_commit_sha: Some(String::from("old")),
            file_count: 1,
            symbol_count: 1,
        };

        assert!(should_refresh_index(&metadata, Some("new")));
        assert!(!should_refresh_index(&metadata, Some("old")));
        assert!(!should_refresh_index(&metadata, None));
    }

    #[test]
    fn stale_indexing_marker_is_removed() {
        let fixture = TempGitFixture::new("rust-fixture");
        let marker_path = indexing_marker_path(fixture.path()).expect("marker path should resolve");
        if let Some(parent) = marker_path.parent() {
            fs::create_dir_all(parent).expect("marker parent should exist");
        }

        fs::write(&marker_path, b"busy").expect("marker file should be created");
        clear_stale_indexing_marker(&marker_path, StdDuration::ZERO);

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
