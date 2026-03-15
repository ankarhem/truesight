pub mod embed;
pub mod incremental;
pub mod indexer;
pub(crate) mod indexing;
pub mod parser;
pub mod repo_context;
pub mod repomap;
pub mod search;
pub(crate) mod util;
pub mod walker;

pub use embed::OnnxEmbedder;
pub use incremental::{ChangeSet, IncrementalIndexer};
pub use indexer::index_repo;
pub use repo_context::{RepoContext, detect_repo_context, detect_repo_context_from_root};
pub use repomap::RepoMapGenerator;
pub use search::SearchEngine;
