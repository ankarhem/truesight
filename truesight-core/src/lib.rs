mod error;
mod index;
mod language;
mod repomap;
mod search;
mod storage;

pub use error::{Result, TruesightError};
pub use index::{
    EmbeddingUpdate, IndexMetadata, IndexStats, IndexStatus, IndexedCodeUnit, IndexedFileRecord,
    PendingEmbedding,
};
pub use language::{CodeUnit, CodeUnitKind, Language};
pub use repomap::{ModuleInfo, RepoMap, SymbolInfo};
pub use search::{MatchType, RankedResult, SearchConfig, SearchResult};
pub use storage::{Embedder, IncrementalStorage, IndexStorage, Storage};

#[cfg(feature = "mocking")]
pub use storage::{MockEmbedder, MockIncrementalStorage, MockIndexStorage, MockStorage};

#[cfg(test)]
mod tests;
