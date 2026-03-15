use std::collections::HashMap;
use std::path::PathBuf;

use rmcp::schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};
use serde::{Deserialize, Serialize};
use truesight_core::RepoMap;

use crate::app::{IndexRepoResponse, IndexRepoStats, SearchRepoResponse};

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct RepoMapRequest {
    #[schemars(description = "Path to the repository root")]
    pub(super) path: String,
    #[schemars(
        description = "Optional repo-relative path prefix to limit the map to a directory or file"
    )]
    pub(super) filter: Option<String>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct SearchRepoRequest {
    #[schemars(description = "Natural language search query")]
    pub(super) query: String,
    #[schemars(description = "Path to the repository root")]
    pub(super) path: String,
    #[schemars(description = "Maximum number of results to return")]
    #[schemars(schema_with = "non_negative_integer_schema")]
    pub(super) limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, JsonSchema)]
pub(super) struct IndexRepoRequest {
    #[schemars(description = "Path to the repository root")]
    pub(super) path: String,
    #[schemars(description = "Force a full reindex instead of incremental refresh")]
    pub(super) full: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(super) struct RepoMapToolResponse {
    pub(super) repo_root: PathBuf,
    pub(super) branch: String,
    pub(super) modules: Vec<ModuleInfoResponse>,
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
pub(super) struct ModuleInfoResponse {
    pub(super) name: String,
    pub(super) path: PathBuf,
    pub(super) files: Vec<String>,
    pub(super) symbols: Vec<SymbolInfoResponse>,
    pub(super) depends_on: Vec<String>,
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
pub(super) struct SymbolInfoResponse {
    pub(super) name: String,
    pub(super) kind: String,
    pub(super) signature: String,
    pub(super) doc: Option<String>,
    pub(super) file: String,
    #[schemars(schema_with = "non_negative_integer_schema")]
    pub(super) line: u32,
}

impl From<truesight_core::SymbolInfo> for SymbolInfoResponse {
    fn from(value: truesight_core::SymbolInfo) -> Self {
        Self {
            name: value.name,
            kind: value.kind.to_string(),
            signature: value.signature,
            doc: value.doc,
            file: value.file,
            line: value.line,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(super) struct SearchRepoToolResponse {
    pub(super) query: String,
    pub(super) results: Vec<SearchResultResponse>,
    #[schemars(schema_with = "non_negative_integer_schema")]
    pub(super) total_results: usize,
    pub(super) search_mode: String,
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
pub(super) struct SearchResultResponse {
    pub(super) kind: String,
    pub(super) name: String,
    pub(super) path: PathBuf,
    #[schemars(schema_with = "non_negative_integer_schema")]
    pub(super) line: u32,
    pub(super) signature: String,
    pub(super) doc: Option<String>,
    pub(super) snippet: String,
    pub(super) score: f32,
    pub(super) match_type: String,
}

impl From<truesight_core::SearchResult> for SearchResultResponse {
    fn from(value: truesight_core::SearchResult) -> Self {
        Self {
            kind: value.kind.to_string(),
            name: value.name,
            path: value.path,
            line: value.line,
            signature: value.signature,
            doc: value.doc,
            snippet: value.snippet,
            score: value.score,
            match_type: value.match_type.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub(super) struct IndexRepoToolResponse {
    pub(super) status: String,
    pub(super) repo_root: PathBuf,
    pub(super) branch: String,
    pub(super) stats: IndexRepoStatsResponse,
    #[schemars(schema_with = "string_to_non_negative_integer_map_schema")]
    pub(super) languages: HashMap<String, u32>,
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
pub(super) struct IndexRepoStatsResponse {
    #[schemars(schema_with = "non_negative_integer_schema")]
    files_scanned: u32,
    #[schemars(schema_with = "non_negative_integer_schema")]
    files_indexed: u32,
    #[schemars(schema_with = "non_negative_integer_schema")]
    files_skipped: u32,
    #[schemars(schema_with = "non_negative_integer_schema")]
    symbols_extracted: u32,
    #[schemars(schema_with = "non_negative_integer_schema")]
    chunks_embedded: u32,
    #[schemars(schema_with = "non_negative_integer_schema")]
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

fn non_negative_integer_schema(_: &mut SchemaGenerator) -> Schema {
    json_schema!({
        "type": "integer",
        "minimum": 0,
    })
}

fn string_to_non_negative_integer_map_schema(_: &mut SchemaGenerator) -> Schema {
    json_schema!({
        "type": "object",
        "additionalProperties": {
            "type": "integer",
            "minimum": 0,
        },
    })
}
