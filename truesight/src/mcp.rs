use std::collections::HashMap;
use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use rmcp::{
    Json, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    schemars::JsonSchema,
    tool, tool_handler, tool_router,
    transport::io::stdio,
};
use serde::{Deserialize, Serialize};
use truesight_core::{CodeUnitKind, MatchType, RepoMap};

use crate::app::{AppServices, IndexRepoResponse, IndexRepoStats, SearchRepoResponse};

type AppFuture<T> = Pin<Box<dyn Future<Output = anyhow::Result<T>> + Send + 'static>>;

trait TruesightApp: Send + Sync {
    fn index_repo(&self, path: PathBuf, full: bool) -> AppFuture<IndexRepoResponse>;
    fn search_repo(
        &self,
        path: PathBuf,
        query: String,
        limit: usize,
    ) -> AppFuture<SearchRepoResponse>;
    fn repo_map(&self, path: PathBuf) -> AppFuture<RepoMap>;
}

impl TruesightApp for AppServices {
    fn index_repo(&self, path: PathBuf, full: bool) -> AppFuture<IndexRepoResponse> {
        let this = self.clone();
        Box::pin(async move { this.index_repo_response(path, full).await })
    }

    fn search_repo(
        &self,
        path: PathBuf,
        query: String,
        limit: usize,
    ) -> AppFuture<SearchRepoResponse> {
        let this = self.clone();
        Box::pin(async move { this.search_repo_response(path, query, limit).await })
    }

    fn repo_map(&self, path: PathBuf) -> AppFuture<RepoMap> {
        let this = self.clone();
        Box::pin(async move { this.repo_map_response(path).await })
    }
}

#[derive(Clone)]
pub struct TruesightMcp {
    tool_router: ToolRouter<Self>,
    app: Arc<dyn TruesightApp>,
}

impl Default for TruesightMcp {
    fn default() -> Self {
        Self::new()
    }
}

impl TruesightMcp {
    pub fn new() -> Self {
        Self::with_app(Arc::new(AppServices::new()))
    }

    fn with_app(app: Arc<dyn TruesightApp>) -> Self {
        Self {
            tool_router: Self::tool_router(),
            app,
        }
    }

    pub async fn serve_stdio(self) -> anyhow::Result<()> {
        let server = self.serve(stdio()).await?;
        server.waiting().await?;
        Ok(())
    }
}

#[tool_router]
impl TruesightMcp {
    #[tool(
        description = "Get a structured repository map with modules, files, symbols, and dependency hints"
    )]
    async fn repo_map(
        &self,
        Parameters(RepoMapRequest { path }): Parameters<RepoMapRequest>,
    ) -> Result<Json<RepoMapToolResponse>, String> {
        let repo_map = self
            .app
            .repo_map(PathBuf::from(path))
            .await
            .map_err(|error| error.to_string())?;
        Ok(Json(repo_map.into()))
    }

    #[tool(description = "Search the repository with hybrid lexical and semantic ranking")]
    async fn search_repo(
        &self,
        Parameters(SearchRepoRequest { query, path, limit }): Parameters<SearchRepoRequest>,
    ) -> Result<Json<SearchRepoToolResponse>, String> {
        let response = self
            .app
            .search_repo(PathBuf::from(path), query, limit.unwrap_or(10))
            .await
            .map_err(|error| error.to_string())?;
        Ok(Json(response.into()))
    }

    #[tool(description = "Index or refresh a repository so search_repo and repo_map stay current")]
    async fn index_repo(
        &self,
        Parameters(IndexRepoRequest { path, full }): Parameters<IndexRepoRequest>,
    ) -> Result<Json<IndexRepoToolResponse>, String> {
        let response = self
            .app
            .index_repo(PathBuf::from(path), full.unwrap_or(false))
            .await
            .map_err(|error| error.to_string())?;
        Ok(Json(response.into()))
    }
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct RepoMapRequest {
    #[schemars(description = "Path to the repository root")]
    path: String,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct SearchRepoRequest {
    #[schemars(description = "Natural language search query")]
    query: String,
    #[schemars(description = "Path to the repository root")]
    path: String,
    #[schemars(description = "Maximum number of results to return")]
    limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct IndexRepoRequest {
    #[schemars(description = "Path to the repository root")]
    path: String,
    #[schemars(description = "Force a full reindex instead of incremental refresh")]
    full: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct RepoMapToolResponse {
    repo_root: PathBuf,
    branch: String,
    modules: Vec<ModuleInfoResponse>,
}

impl From<RepoMap> for RepoMapToolResponse {
    fn from(value: RepoMap) -> Self {
        Self {
            repo_root: value.repo_root,
            branch: value.branch,
            modules: value.modules.into_iter().map(Into::into).collect(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct ModuleInfoResponse {
    name: String,
    path: PathBuf,
    files: Vec<String>,
    symbols: Vec<SymbolInfoResponse>,
    depends_on: Vec<String>,
}

impl From<truesight_core::ModuleInfo> for ModuleInfoResponse {
    fn from(value: truesight_core::ModuleInfo) -> Self {
        Self {
            name: value.name,
            path: value.path,
            files: value.files,
            symbols: value.symbols.into_iter().map(Into::into).collect(),
            depends_on: value.depends_on,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SymbolInfoResponse {
    name: String,
    kind: String,
    signature: String,
    doc: Option<String>,
    file: String,
    line: u32,
}

impl From<truesight_core::SymbolInfo> for SymbolInfoResponse {
    fn from(value: truesight_core::SymbolInfo) -> Self {
        Self {
            name: value.name,
            kind: kind_label(value.kind).to_string(),
            signature: value.signature,
            doc: value.doc,
            file: value.file,
            line: value.line,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SearchRepoToolResponse {
    query: String,
    results: Vec<SearchResultResponse>,
    total_results: usize,
    search_mode: String,
}

impl From<SearchRepoResponse> for SearchRepoToolResponse {
    fn from(value: SearchRepoResponse) -> Self {
        Self {
            query: value.query,
            results: value.results.into_iter().map(Into::into).collect(),
            total_results: value.total_results,
            search_mode: value.search_mode,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct SearchResultResponse {
    kind: String,
    name: String,
    path: PathBuf,
    line: u32,
    signature: String,
    doc: Option<String>,
    snippet: String,
    score: f32,
    match_type: String,
}

impl From<truesight_core::SearchResult> for SearchResultResponse {
    fn from(value: truesight_core::SearchResult) -> Self {
        Self {
            kind: kind_label(value.kind).to_string(),
            name: value.name,
            path: value.path,
            line: value.line,
            signature: value.signature,
            doc: value.doc,
            snippet: value.snippet,
            score: value.score,
            match_type: match_type_label(value.match_type).to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct IndexRepoToolResponse {
    status: String,
    repo_root: PathBuf,
    branch: String,
    stats: IndexRepoStatsResponse,
    languages: HashMap<String, u32>,
}

impl From<IndexRepoResponse> for IndexRepoToolResponse {
    fn from(value: IndexRepoResponse) -> Self {
        Self {
            status: value.status,
            repo_root: value.repo_root,
            branch: value.branch,
            stats: value.stats.into(),
            languages: value.languages,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct IndexRepoStatsResponse {
    files_scanned: u32,
    files_indexed: u32,
    files_skipped: u32,
    symbols_extracted: u32,
    chunks_embedded: u32,
    duration_ms: u64,
}

impl From<IndexRepoStats> for IndexRepoStatsResponse {
    fn from(value: IndexRepoStats) -> Self {
        Self {
            files_scanned: value.files_scanned,
            files_indexed: value.files_indexed,
            files_skipped: value.files_skipped,
            symbols_extracted: value.symbols_extracted,
            chunks_embedded: value.chunks_embedded,
            duration_ms: value.duration_ms,
        }
    }
}

#[tool_handler]
impl ServerHandler for TruesightMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            "Truesight indexes local repositories and exposes exactly three tools: index_repo, search_repo, and repo_map.",
        )
    }
}

pub async fn run() -> anyhow::Result<()> {
    TruesightMcp::new().serve_stdio().await
}

fn kind_label(kind: CodeUnitKind) -> &'static str {
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

fn match_type_label(match_type: MatchType) -> &'static str {
    match match_type {
        MatchType::Fts => "fts",
        MatchType::Vector => "vector",
        MatchType::Hybrid => "hybrid",
    }
}

#[cfg(test)]
mod tests {
    use std::future::Future;
    use std::pin::Pin;

    use super::*;
    use truesight_core::{ModuleInfo, SearchResult, SymbolInfo};

    #[derive(Clone, Default)]
    struct FakeApp;

    impl TruesightApp for FakeApp {
        fn index_repo(
            &self,
            path: PathBuf,
            _full: bool,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<IndexRepoResponse>> + Send + 'static>>
        {
            Box::pin(async move {
                Ok(IndexRepoResponse {
                    status: String::from("completed"),
                    repo_root: path,
                    branch: String::from("main"),
                    stats: IndexRepoStats {
                        files_scanned: 2,
                        files_indexed: 2,
                        files_skipped: 0,
                        symbols_extracted: 3,
                        chunks_embedded: 3,
                        duration_ms: 12,
                    },
                    languages: HashMap::from([(String::from("rust"), 2)]),
                })
            })
        }

        fn search_repo(
            &self,
            path: PathBuf,
            query: String,
            _limit: usize,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<SearchRepoResponse>> + Send + 'static>>
        {
            Box::pin(async move {
                Ok(SearchRepoResponse {
                    query,
                    total_results: 1,
                    results: vec![SearchResult {
                        kind: CodeUnitKind::Function,
                        name: String::from("retry_payment"),
                        path,
                        line: 23,
                        signature: String::from("pub fn retry_payment() -> bool"),
                        doc: Some(String::from("Retries failed payments.")),
                        snippet: String::from("pub fn retry_payment() -> bool { true }"),
                        score: 1.0,
                        match_type: MatchType::Hybrid,
                    }],
                    search_mode: String::from("hybrid"),
                    repo_root: PathBuf::from("/repo"),
                    branch: String::from("main"),
                })
            })
        }

        fn repo_map(
            &self,
            path: PathBuf,
        ) -> Pin<Box<dyn Future<Output = anyhow::Result<RepoMap>> + Send + 'static>> {
            Box::pin(async move {
                Ok(RepoMap {
                    repo_root: path,
                    branch: String::from("main"),
                    modules: vec![ModuleInfo {
                        name: String::from("src"),
                        path: PathBuf::from("src"),
                        files: vec![String::from("lib.rs")],
                        symbols: vec![SymbolInfo {
                            name: String::from("AuthService"),
                            kind: CodeUnitKind::Struct,
                            signature: String::from("pub struct AuthService"),
                            doc: Some(String::from("Service.")),
                            file: String::from("lib.rs"),
                            line: 1,
                        }],
                        depends_on: vec![],
                    }],
                })
            })
        }
    }

    #[test]
    fn tool_router_lists_exactly_three_planned_tools() {
        let server = TruesightMcp::with_app(Arc::new(FakeApp));
        let mut names = server
            .tool_router
            .list_all()
            .into_iter()
            .map(|tool| tool.name.to_string())
            .collect::<Vec<_>>();
        names.sort_unstable();

        assert_eq!(names, vec!["index_repo", "repo_map", "search_repo"]);
    }

    #[tokio::test]
    async fn tool_methods_return_structured_contract_data() {
        let server = TruesightMcp::with_app(Arc::new(FakeApp));

        let index_result = server
            .index_repo(Parameters(IndexRepoRequest {
                path: String::from("/repo"),
                full: Some(true),
            }))
            .await
            .unwrap()
            .0;
        assert_eq!(index_result.status, "completed");
        assert_eq!(index_result.languages.get("rust"), Some(&2));

        let search_result = server
            .search_repo(Parameters(SearchRepoRequest {
                query: String::from("retry"),
                path: String::from("src/retry.rs"),
                limit: Some(5),
            }))
            .await
            .unwrap()
            .0;
        assert_eq!(search_result.query, "retry");
        assert_eq!(search_result.results[0].name, "retry_payment");
        assert_eq!(search_result.results[0].match_type, "hybrid");

        let repo_map_result = server
            .repo_map(Parameters(RepoMapRequest {
                path: String::from("/repo"),
            }))
            .await
            .unwrap()
            .0;
        assert_eq!(repo_map_result.modules[0].symbols[0].name, "AuthService");
        assert_eq!(repo_map_result.modules[0].symbols[0].kind, "struct");
    }

    #[test]
    fn server_info_enables_tool_capabilities() {
        let info = TruesightMcp::with_app(Arc::new(FakeApp)).get_info();

        assert!(info.capabilities.tools.is_some());
        assert!(info.instructions.unwrap_or_default().contains("index_repo"));
    }
}
