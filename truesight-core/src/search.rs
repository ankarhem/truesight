use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::{CodeUnit, CodeUnitKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum MatchType {
    Fts,
    Vector,
    Hybrid,
}

impl std::fmt::Display for MatchType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatchType::Fts => write!(f, "fts"),
            MatchType::Vector => write!(f, "vector"),
            MatchType::Hybrid => write!(f, "hybrid"),
        }
    }
}

impl std::str::FromStr for MatchType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "fts" => Ok(MatchType::Fts),
            "vector" => Ok(MatchType::Vector),
            "hybrid" => Ok(MatchType::Hybrid),
            other => Err(format!("unknown match type: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResult {
    pub kind: CodeUnitKind,
    pub name: String,
    pub path: PathBuf,
    pub line: u32,
    pub signature: String,
    pub doc: Option<String>,
    pub snippet: String,
    pub score: f32,
    pub match_type: MatchType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    pub unit: CodeUnit,
    pub fts_score: Option<f32>,
    pub vector_score: Option<f32>,
    pub combined_score: f32,
    pub match_type: MatchType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub limit: usize,
    pub rrf_k: u32,
    pub use_fts: bool,
    pub use_vector: bool,
    pub min_score: f32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            limit: 10,
            rrf_k: 60,
            use_fts: true,
            use_vector: true,
            min_score: 0.0,
        }
    }
}
