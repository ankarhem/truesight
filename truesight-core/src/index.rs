use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::{CodeUnit, Language};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexStats {
    pub files_scanned: u32,
    pub files_indexed: u32,
    pub files_skipped: u32,
    pub symbols_extracted: u32,
    pub chunks_embedded: u32,
    pub duration_ms: u64,
    pub languages: HashMap<Language, u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IndexStatus {
    Indexing,
    Ready,
    Failed,
}

impl Default for IndexStatus {
    fn default() -> Self {
        Self::Ready
    }
}

impl std::fmt::Display for IndexStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexStatus::Indexing => write!(f, "indexing"),
            IndexStatus::Ready => write!(f, "ready"),
            IndexStatus::Failed => write!(f, "failed"),
        }
    }
}

impl std::str::FromStr for IndexStatus {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "indexing" => Ok(IndexStatus::Indexing),
            "ready" => Ok(IndexStatus::Ready),
            "failed" => Ok(IndexStatus::Failed),
            other => Err(format!("unknown index status: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    pub repo_id: String,
    pub branch: String,
    pub status: IndexStatus,
    pub last_indexed_at: chrono::DateTime<chrono::Utc>,
    pub last_commit_sha: Option<String>,
    pub last_seen_commit_sha: Option<String>,
    pub file_count: u32,
    pub symbol_count: u32,
    pub embedding_model: String,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexedCodeUnit {
    pub unit: CodeUnit,
    pub embedding: Option<Vec<f32>>,
    pub file_hash: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexedFileRecord {
    pub file_path: PathBuf,
    pub file_hash: String,
    pub chunk_count: u32,
    pub indexed_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingEmbedding {
    pub id: String,
    pub signature: String,
    pub doc: Option<String>,
    pub content: String,
}

impl PendingEmbedding {
    pub fn embedding_text(&self) -> String {
        match self.doc.as_deref() {
            Some(doc) if !doc.is_empty() => {
                format!("{} {} {}", self.signature, doc, self.content)
            }
            _ => format!("{} {}", self.signature, self.content),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUpdate {
    pub id: String,
    pub embedding: Vec<f32>,
}
