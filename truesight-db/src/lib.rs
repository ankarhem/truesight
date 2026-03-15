use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use libsql::{Connection, Database as LibsqlDatabase, params};
use sha2::{Digest, Sha256};
use tracing::warn;
use truesight_core::{
    CodeUnit, EmbeddingUpdate, IncrementalStorage, IndexMetadata, IndexStatus, IndexStorage,
    IndexedCodeUnit, IndexedFileRecord, MatchType, PendingEmbedding, RankedResult, Result, Storage,
    TruesightError,
};

mod bookkeeping;
mod error;
mod helpers;
mod lifecycle;
mod rows;
mod schema;
mod search;
mod search_ops;
mod storage_impl;

use error::DatabaseError;
use helpers::{
    code_unit_id, cosine_similarity, decode_embedding, default_file_hash, encode_embedding,
    hybrid_candidate_limit, normalize_fts_score, normalize_repo_root, parse_timestamp,
    path_to_string, sanitize_fts_query, sanitize_path_segment, short_hex_digest,
};
use rows::{code_unit_from_row, insert_code_unit};
use schema::{MIGRATION_TABLE_SQL, ensure_vector_index, run_incremental_migrations};
use search::{fuse_ranked_results, ranked_result_from_row};

#[derive(Clone)]
pub struct Database {
    db: Arc<LibsqlDatabase>,
}

#[cfg(test)]
mod tests;
