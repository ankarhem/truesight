use std::future::Future;
use std::path::PathBuf;
use std::pin::Pin;
use std::sync::Arc;

use rmcp::{
    Json, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{ServerCapabilities, ServerInfo},
    tool, tool_handler, tool_router,
    transport::io::stdio,
};
use truesight_core::RepoMap;

use crate::app::{AppServices, IndexRepoResponse, SearchRepoResponse};

mod types;

use types::{
    IndexRepoRequest, IndexRepoToolResponse, RepoMapRequest, RepoMapToolResponse,
    SearchRepoRequest, SearchRepoToolResponse,
};

type AppFuture<T> = Pin<Box<dyn Future<Output = anyhow::Result<T>> + Send + 'static>>;

#[cfg_attr(test, mockall::automock)]
trait TruesightApp: Send + Sync {
    fn index_repo(&self, path: PathBuf, full: bool) -> AppFuture<IndexRepoResponse>;
    fn search_repo(
        &self,
        path: PathBuf,
        query: String,
        limit: usize,
    ) -> AppFuture<SearchRepoResponse>;
    fn repo_map(&self, path: PathBuf, filter: Option<String>) -> AppFuture<RepoMap>;
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

    fn repo_map(&self, path: PathBuf, filter: Option<String>) -> AppFuture<RepoMap> {
        let this = self.clone();
        Box::pin(async move { this.repo_map_response(path, filter).await })
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
        description = "Get a structured repository map after search narrows the area; best for module boundaries, key symbols, dependency hints, and optionally focusing on a repo-relative path prefix"
    )]
    async fn repo_map(
        &self,
        Parameters(RepoMapRequest { path, filter }): Parameters<RepoMapRequest>,
    ) -> Result<Json<RepoMapToolResponse>, String> {
        let repo_map = self
            .app
            .repo_map(PathBuf::from(path), filter)
            .await
            .map_err(|error| error.to_string())?;
        Ok(Json(repo_map.into()))
    }

    #[tool(
        description = "Search when you do not know the exact location yet; best for ranked symbol lookup, semantic code discovery, and pairing with grep for exact text checks"
    )]
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

    #[tool(
        description = "Refresh repository search data when results are missing, stale, or after major repository changes; use this to repair or warm the index for search_repo and repo_map"
    )]
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

#[tool_handler]
impl ServerHandler for TruesightMcp {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build()).with_instructions(
            r#"
| Tool | Description |
|------|-------------|
| `search_repo` | Default first step for exploratory codebase work when the exact file or symbol is unknown; ranked lexical + semantic search |
| `repo_map` | Use after search when you need module boundaries, key symbols, and dependency context; pass `filter` to focus on a repo-relative directory or file |
| `index_repo` | Refresh or repair the index when search results are missing, stale, or after major repository changes |

Recommended agent workflow:

1. Start with `search_repo` when the location is unknown or you want likely implementations ranked for you.
2. Use `repo_map` once search has narrowed the area and you want structural context; add `filter` when you already know the relevant directory or file.
3. Use `grep` alongside Truesight when you need exact strings, regex matches, or exhaustive literal confirmation.
4. Use `index_repo` only when the repository changed substantially or search data needs a refresh."#,
        )
    }
}

pub async fn run() -> anyhow::Result<()> {
    TruesightMcp::new().serve_stdio().await
}

#[cfg(test)]
mod tests {
    use super::*;
    use truesight_core::{CodeUnitKind, MatchType, ModuleInfo, SearchResult, SymbolInfo};

    fn expect_index_repo(app: &mut MockTruesightApp) {
        app.expect_index_repo()
            .times(1)
            .withf(|path, full| path == &PathBuf::from("/repo") && *full)
            .returning(|path, _| {
                Box::pin(async move {
                    Ok(IndexRepoResponse {
                        status: String::from("completed"),
                        repo_root: path,
                        branch: String::from("main"),
                        stats: crate::app::IndexRepoStats {
                            files_scanned: 2,
                            files_indexed: 2,
                            files_skipped: 0,
                            symbols_extracted: 3,
                            chunks_embedded: 3,
                            duration_ms: 12,
                        },
                        languages: std::collections::HashMap::from([(String::from("rust"), 2)]),
                    })
                })
            });
    }

    fn expect_search_repo(app: &mut MockTruesightApp) {
        app.expect_search_repo()
            .times(1)
            .withf(|path, query, limit| {
                path == &PathBuf::from("src/retry.rs") && query == "retry" && *limit == 5
            })
            .returning(|path, query, _| {
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
            });
    }

    fn expect_repo_map(app: &mut MockTruesightApp) {
        app.expect_repo_map()
            .times(1)
            .withf(|path, filter| {
                path == &PathBuf::from("/repo") && filter.as_deref() == Some("src")
            })
            .returning(|path, _| {
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
            });
    }

    #[test]
    fn tool_router_lists_exactly_three_planned_tools() {
        let server = TruesightMcp::with_app(Arc::new(MockTruesightApp::new()));
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
        let mut app = MockTruesightApp::new();
        expect_index_repo(&mut app);
        expect_search_repo(&mut app);
        expect_repo_map(&mut app);
        let server = TruesightMcp::with_app(Arc::new(app));

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
                filter: Some(String::from("src")),
            }))
            .await
            .unwrap()
            .0;
        assert_eq!(repo_map_result.modules[0].symbols[0].name, "AuthService");
        assert_eq!(repo_map_result.modules[0].symbols[0].kind, "struct");
    }

    #[test]
    fn server_info_enables_tool_capabilities() {
        let info = TruesightMcp::with_app(Arc::new(MockTruesightApp::new())).get_info();

        assert!(info.capabilities.tools.is_some());
        assert!(info.instructions.unwrap_or_default().contains("index_repo"));
    }
}
