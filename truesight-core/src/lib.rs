//! Core types, traits, and error definitions for Truesight.
//!
//! This crate contains only shared contracts - no implementation logic.

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// =============================================================================
// Error Types
// =============================================================================

/// The unified error type for all Truesight operations.
#[derive(Debug, thiserror::Error)]
pub enum TruesightError {
    #[error("Parse error: {0}")]
    Parse(String),

    #[error("Database error: {0}")]
    Database(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Index error: {0}")]
    Index(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),

    #[error("Git error: {0}")]
    Git(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// Convenience type alias for Results in Truesight.
pub type Result<T> = std::result::Result<T, TruesightError>;

// =============================================================================
// Language and Code Unit Types
// =============================================================================

/// Supported programming languages for parsing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    TypeScript,
    CSharp,
}

impl Language {
    /// Returns the file extensions associated with this language.
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["rs"],
            Language::TypeScript => &["ts", "tsx"],
            Language::CSharp => &["cs"],
        }
    }

    /// Detect the programming language from a file path's extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("rs") => Some(Language::Rust),
            Some("ts") | Some("tsx") => Some(Language::TypeScript),
            Some("cs") => Some(Language::CSharp),
            _ => None,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Rust => write!(f, "rust"),
            Language::TypeScript => write!(f, "typescript"),
            Language::CSharp => write!(f, "csharp"),
        }
    }
}

impl std::str::FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "rust" => Ok(Language::Rust),
            "typescript" => Ok(Language::TypeScript),
            "csharp" => Ok(Language::CSharp),
            other => Err(format!("unknown language: {other}")),
        }
    }
}

/// The kind of code unit extracted from source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum CodeUnitKind {
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Class,
    Interface,
    Constant,
    Module,
}

impl std::fmt::Display for CodeUnitKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeUnitKind::Function => write!(f, "function"),
            CodeUnitKind::Method => write!(f, "method"),
            CodeUnitKind::Struct => write!(f, "struct"),
            CodeUnitKind::Enum => write!(f, "enum"),
            CodeUnitKind::Trait => write!(f, "trait"),
            CodeUnitKind::Class => write!(f, "class"),
            CodeUnitKind::Interface => write!(f, "interface"),
            CodeUnitKind::Constant => write!(f, "constant"),
            CodeUnitKind::Module => write!(f, "module"),
        }
    }
}

impl std::str::FromStr for CodeUnitKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "function" => Ok(CodeUnitKind::Function),
            "method" => Ok(CodeUnitKind::Method),
            "struct" => Ok(CodeUnitKind::Struct),
            "enum" => Ok(CodeUnitKind::Enum),
            "trait" => Ok(CodeUnitKind::Trait),
            "class" => Ok(CodeUnitKind::Class),
            "interface" => Ok(CodeUnitKind::Interface),
            "constant" => Ok(CodeUnitKind::Constant),
            "module" => Ok(CodeUnitKind::Module),
            other => Err(format!("unknown code unit kind: {other}")),
        }
    }
}

/// A code unit extracted from parsing source files.
///
/// This represents a single symbol (function, class, struct, etc.) with its
/// full context including signature, documentation, and source content.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CodeUnit {
    /// The name of the symbol (e.g., function name, class name).
    pub name: String,
    /// The kind of code unit (function, struct, class, etc.).
    pub kind: CodeUnitKind,
    /// The full signature (e.g., function parameters and return type, without body).
    pub signature: String,
    /// Doc comments associated with this symbol, if any.
    pub doc: Option<String>,
    /// Absolute or relative path to the source file.
    pub file_path: PathBuf,
    /// Starting line number (1-indexed).
    pub line_start: u32,
    /// Ending line number (1-indexed, inclusive).
    pub line_end: u32,
    /// Full source text of this code unit.
    pub content: String,
    /// Parent symbol name (e.g., containing class/struct/impl).
    pub parent: Option<String>,
    /// The programming language of this code unit.
    pub language: Language,
}

// =============================================================================
// Search Types
// =============================================================================

/// The type of match that produced a search result.
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

/// A search result returned from the search engine.
///
/// This is the type returned by MCP `search_repo` tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResult {
    /// The kind of code unit matched.
    pub kind: CodeUnitKind,
    /// The name of the matched symbol.
    pub name: String,
    /// Path to the file containing the match.
    pub path: PathBuf,
    /// Line number where the symbol starts (1-indexed).
    pub line: u32,
    /// The full signature of the matched symbol.
    pub signature: String,
    /// Doc comments, if available.
    pub doc: Option<String>,
    /// A snippet of the source code around the match.
    pub snippet: String,
    /// Relevance score (0.0 to 1.0, higher is better).
    pub score: f32,
    /// How this result was matched (FTS, vector, or hybrid).
    pub match_type: MatchType,
}

/// Internal result type used by the search engine before converting to SearchResult.
///
/// Contains the code unit plus search metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankedResult {
    /// The code unit that matched.
    pub unit: CodeUnit,
    /// BM25 score from FTS5 (if applicable).
    pub fts_score: Option<f32>,
    /// Vector similarity score (if applicable).
    pub vector_score: Option<f32>,
    /// Combined RRF score.
    pub combined_score: f32,
    /// How this result was matched.
    pub match_type: MatchType,
}

/// Configuration for search behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Maximum number of results to return.
    pub limit: usize,
    /// RRF constant K for score fusion.
    pub rrf_k: u32,
    /// Whether to use full-text search.
    pub use_fts: bool,
    /// Whether to use vector/semantic search.
    pub use_vector: bool,
    /// Minimum score threshold for results.
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

// =============================================================================
// Repo Map Types
// =============================================================================

/// Information about a single symbol in the repo map.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SymbolInfo {
    /// The name of the symbol.
    pub name: String,
    /// The kind of symbol (function, struct, etc.).
    pub kind: CodeUnitKind,
    /// The full signature.
    pub signature: String,
    /// Doc comments, if available.
    pub doc: Option<String>,
    /// The file containing this symbol (relative to module path).
    pub file: String,
    /// Line number where the symbol starts (1-indexed).
    pub line: u32,
}

/// Information about a module/directory in the repository.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModuleInfo {
    /// The name of the module (typically directory name).
    pub name: String,
    /// Path to the module directory.
    pub path: PathBuf,
    /// Files in this module.
    pub files: Vec<String>,
    /// Symbols defined in this module.
    pub symbols: Vec<SymbolInfo>,
    /// Names of modules this module depends on.
    pub depends_on: Vec<String>,
}

/// The complete repository map returned by `repo_map` MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RepoMap {
    /// Root path of the repository.
    pub repo_root: PathBuf,
    /// Current branch name.
    pub branch: String,
    /// Modules in the repository.
    pub modules: Vec<ModuleInfo>,
}

// =============================================================================
// Index Types
// =============================================================================

/// Statistics from an indexing operation.
///
/// This is part of the response from the `index_repo` MCP tool.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct IndexStats {
    /// Number of files examined.
    pub files_scanned: u32,
    /// Number of files successfully indexed.
    pub files_indexed: u32,
    /// Number of files skipped (filtered out).
    pub files_skipped: u32,
    /// Number of symbols extracted.
    pub symbols_extracted: u32,
    /// Number of code chunks embedded.
    pub chunks_embedded: u32,
    /// Time taken in milliseconds.
    pub duration_ms: u64,
    /// Count of files per language.
    pub languages: HashMap<Language, u32>,
}

/// Metadata about an indexed repository/branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Repository identifier (absolute path).
    pub repo_id: String,
    /// Branch name.
    pub branch: String,
    /// When this index was last updated.
    pub last_indexed_at: chrono::DateTime<chrono::Utc>,
    /// Git commit SHA at time of indexing (if applicable).
    pub last_commit_sha: Option<String>,
    /// Number of files in the index.
    pub file_count: u32,
    /// Number of symbols in the index.
    pub symbol_count: u32,
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
    pub indexed_at: chrono::DateTime<chrono::Utc>,
}

// =============================================================================
// Traits
// =============================================================================

/// Trait for storage backends that persist and query code units.
#[async_trait]
pub trait Storage: Send + Sync {
    /// Store code units for a repository/branch.
    async fn store_code_units(&self, repo_id: &str, branch: &str, units: &[CodeUnit])
    -> Result<()>;

    /// Full-text search for code units.
    async fn search_fts(
        &self,
        repo_id: &str,
        branch: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<RankedResult>>;

    /// Vector similarity search for code units.
    async fn search_vector(
        &self,
        repo_id: &str,
        branch: &str,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<RankedResult>>;

    /// Get metadata for an indexed repository/branch.
    async fn get_index_metadata(
        &self,
        repo_id: &str,
        branch: &str,
    ) -> Result<Option<IndexMetadata>>;

    /// Set/update metadata for an indexed repository/branch.
    async fn set_index_metadata(
        &self,
        repo_id: &str,
        branch: &str,
        meta: &IndexMetadata,
    ) -> Result<()>;

    /// Delete all index data for a repository/branch.
    async fn delete_branch_index(&self, repo_id: &str, branch: &str) -> Result<()>;

    /// Get all code units for a repository/branch (for repo_map generation).
    async fn get_all_symbols(&self, repo_id: &str, branch: &str) -> Result<Vec<CodeUnit>>;
}

#[async_trait]
pub trait IndexStorage: Storage {
    async fn store_indexed_units(
        &self,
        repo_id: &str,
        branch: &str,
        units: &[IndexedCodeUnit],
    ) -> Result<()>;

    async fn upsert_indexed_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
        file_hash: &str,
    ) -> Result<()>;

    async fn replace_branch_index(
        &self,
        repo_id: &str,
        branch: &str,
        units: &[IndexedCodeUnit],
        indexed_files: &[IndexedFileRecord],
        metadata: &IndexMetadata,
    ) -> Result<()> {
        self.delete_branch_index(repo_id, branch).await?;

        if !units.is_empty() {
            self.store_indexed_units(repo_id, branch, units).await?;
        }

        for file in indexed_files {
            self.upsert_indexed_file(repo_id, branch, &file.file_path, &file.file_hash)
                .await?;
        }

        self.set_index_metadata(repo_id, branch, metadata).await
    }
}

#[async_trait]
pub trait IncrementalStorage: IndexStorage {
    async fn get_indexed_files(
        &self,
        repo_id: &str,
        branch: &str,
    ) -> Result<Vec<IndexedFileRecord>>;

    async fn delete_units_for_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
    ) -> Result<()>;

    async fn delete_indexed_file(
        &self,
        repo_id: &str,
        branch: &str,
        file_path: &Path,
    ) -> Result<()>;
}

/// Trait for embedding text into dense vectors.
pub trait Embedder: Send + Sync {
    /// Embed a single text string.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed multiple text strings in batch.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Get the dimension of the embedding vectors.
    fn dimension(&self) -> usize;
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // CodeUnit Tests
    // =========================================================================

    #[test]
    fn code_unit_serializes_to_expected_json_fields() {
        let unit = CodeUnit {
            name: "validate_token".to_string(),
            kind: CodeUnitKind::Function,
            signature: "pub fn validate_token(token: &str) -> Result<Claims, AuthError>".to_string(),
            doc: Some("Validates a JWT token and returns claims".to_string()),
            file_path: PathBuf::from("src/auth/jwt_validator.rs"),
            line_start: 42,
            line_end: 55,
            content: "pub fn validate_token(token: &str) -> Result<Claims, AuthError> {\n    // implementation\n}".to_string(),
            parent: None,
            language: Language::Rust,
        };

        let json = serde_json::to_string(&unit).expect("Failed to serialize CodeUnit");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse JSON");

        // Verify all expected fields exist with correct values
        assert_eq!(parsed["name"], "validate_token");
        assert_eq!(parsed["kind"], "function");
        assert_eq!(
            parsed["signature"],
            "pub fn validate_token(token: &str) -> Result<Claims, AuthError>"
        );
        assert_eq!(parsed["doc"], "Validates a JWT token and returns claims");
        assert_eq!(parsed["file_path"], "src/auth/jwt_validator.rs");
        assert_eq!(parsed["line_start"], 42);
        assert_eq!(parsed["line_end"], 55);
        assert_eq!(parsed["language"], "rust");
        assert_eq!(parsed["parent"], serde_json::Value::Null);
    }

    #[test]
    fn code_unit_round_trips_through_json() {
        let original = CodeUnit {
            name: "AuthService".to_string(),
            kind: CodeUnitKind::Struct,
            signature: "pub struct AuthService { db: DbPool }".to_string(),
            doc: None,
            file_path: PathBuf::from("src/auth/mod.rs"),
            line_start: 15,
            line_end: 20,
            content: "pub struct AuthService { db: DbPool }".to_string(),
            parent: Some("auth".to_string()),
            language: Language::Rust,
        };

        let json = serde_json::to_string(&original).expect("Failed to serialize");
        let deserialized: CodeUnit = serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized.name, original.name);
        assert_eq!(deserialized.kind, original.kind);
        assert_eq!(deserialized.signature, original.signature);
        assert_eq!(deserialized.doc, original.doc);
        assert_eq!(deserialized.file_path, original.file_path);
        assert_eq!(deserialized.line_start, original.line_start);
        assert_eq!(deserialized.line_end, original.line_end);
        assert_eq!(deserialized.content, original.content);
        assert_eq!(deserialized.parent, original.parent);
        assert_eq!(deserialized.language, original.language);
    }

    #[test]
    fn code_unit_kind_serializes_to_lowercase() {
        let kinds = vec![
            (CodeUnitKind::Function, "function"),
            (CodeUnitKind::Method, "method"),
            (CodeUnitKind::Struct, "struct"),
            (CodeUnitKind::Enum, "enum"),
            (CodeUnitKind::Trait, "trait"),
            (CodeUnitKind::Class, "class"),
            (CodeUnitKind::Interface, "interface"),
            (CodeUnitKind::Constant, "constant"),
            (CodeUnitKind::Module, "module"),
        ];

        for (kind, expected) in kinds {
            let json = serde_json::to_string(&kind).expect("Failed to serialize CodeUnitKind");
            assert_eq!(
                json.trim_matches('"'),
                expected,
                "CodeUnitKind::{:?} should serialize to {}",
                kind,
                expected
            );
        }
    }

    #[test]
    fn language_extensions_are_correct() {
        assert_eq!(Language::Rust.extensions(), &["rs"]);
        assert_eq!(Language::TypeScript.extensions(), &["ts", "tsx"]);
        assert_eq!(Language::CSharp.extensions(), &["cs"]);
    }

    // =========================================================================
    // SearchResult Tests
    // =========================================================================

    #[test]
    fn search_result_serializes_to_mcp_contract() {
        let result = SearchResult {
            kind: CodeUnitKind::Function,
            name: "retry_payment".to_string(),
            path: PathBuf::from("src/payments/retry_policy.rs"),
            line: 23,
            signature: "pub async fn retry_payment(payment_id: Uuid, max_retries: u32) -> Result<Payment>".to_string(),
            doc: Some("Retries failed payments with exponential backoff".to_string()),
            snippet: "pub async fn retry_payment(...) {\n    for attempt in 0..max_retries {\n        ...\n    }\n}".to_string(),
            score: 0.87,
            match_type: MatchType::Hybrid,
        };

        let json = serde_json::to_string(&result).expect("Failed to serialize SearchResult");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse JSON");

        // Verify MCP contract fields
        assert_eq!(parsed["kind"], "function");
        assert_eq!(parsed["name"], "retry_payment");
        assert_eq!(parsed["path"], "src/payments/retry_policy.rs");
        assert_eq!(parsed["line"], 23);
        assert_eq!(
            parsed["signature"],
            "pub async fn retry_payment(payment_id: Uuid, max_retries: u32) -> Result<Payment>"
        );
        assert_eq!(
            parsed["doc"],
            "Retries failed payments with exponential backoff"
        );
        assert!(
            parsed["snippet"]
                .as_str()
                .unwrap()
                .contains("retry_payment")
        );
        assert!((parsed["score"].as_f64().unwrap() - 0.87).abs() < 0.001);
        assert_eq!(parsed["match_type"], "hybrid");
    }

    #[test]
    fn search_result_handles_missing_doc() {
        let result = SearchResult {
            kind: CodeUnitKind::Struct,
            name: "Payment".to_string(),
            path: PathBuf::from("src/models/payment.rs"),
            line: 10,
            signature: "pub struct Payment { id: Uuid }".to_string(),
            doc: None,
            snippet: "pub struct Payment { id: Uuid }".to_string(),
            score: 0.65,
            match_type: MatchType::Fts,
        };

        let json = serde_json::to_string(&result).expect("Failed to serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse");

        assert_eq!(parsed["doc"], serde_json::Value::Null);
    }

    #[test]
    fn match_type_serializes_correctly() {
        assert_eq!(
            serde_json::to_string(&MatchType::Fts)
                .unwrap()
                .trim_matches('"'),
            "fts"
        );
        assert_eq!(
            serde_json::to_string(&MatchType::Vector)
                .unwrap()
                .trim_matches('"'),
            "vector"
        );
        assert_eq!(
            serde_json::to_string(&MatchType::Hybrid)
                .unwrap()
                .trim_matches('"'),
            "hybrid"
        );
    }

    // =========================================================================
    // RepoMap Tests
    // =========================================================================

    #[test]
    fn repo_map_serializes_to_mcp_contract() {
        let repo_map = RepoMap {
            repo_root: PathBuf::from("/path/to/repo"),
            branch: "main".to_string(),
            modules: vec![ModuleInfo {
                name: "auth".to_string(),
                path: PathBuf::from("src/auth"),
                files: vec![
                    "auth_service.rs".to_string(),
                    "jwt_validator.rs".to_string(),
                ],
                symbols: vec![
                    SymbolInfo {
                        name: "AuthService".to_string(),
                        kind: CodeUnitKind::Struct,
                        signature: "pub struct AuthService { ... }".to_string(),
                        doc: Some("Handles authentication".to_string()),
                        file: "auth_service.rs".to_string(),
                        line: 15,
                    },
                    SymbolInfo {
                        name: "validate_token".to_string(),
                        kind: CodeUnitKind::Function,
                        signature: "pub fn validate_token(token: &str) -> Result<Claims>"
                            .to_string(),
                        doc: Some("Validates a JWT token".to_string()),
                        file: "jwt_validator.rs".to_string(),
                        line: 42,
                    },
                ],
                depends_on: vec!["db".to_string(), "config".to_string()],
            }],
        };

        let json = serde_json::to_string(&repo_map).expect("Failed to serialize RepoMap");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse JSON");

        // Verify MCP contract top-level fields
        assert_eq!(parsed["repo_root"], "/path/to/repo");
        assert_eq!(parsed["branch"], "main");
        assert!(parsed["modules"].is_array());

        let module = &parsed["modules"][0];
        assert_eq!(module["name"], "auth");
        assert_eq!(module["path"], "src/auth");
        assert!(module["files"].is_array());
        assert!(module["symbols"].is_array());
        assert!(module["depends_on"].is_array());

        // Verify symbol structure
        let symbol = &module["symbols"][0];
        assert_eq!(symbol["name"], "AuthService");
        assert_eq!(symbol["kind"], "struct");
        assert_eq!(symbol["signature"], "pub struct AuthService { ... }");
        assert_eq!(symbol["doc"], "Handles authentication");
        assert_eq!(symbol["file"], "auth_service.rs");
        assert_eq!(symbol["line"], 15);
    }

    #[test]
    fn symbol_info_handles_missing_doc() {
        let symbol = SymbolInfo {
            name: "helper".to_string(),
            kind: CodeUnitKind::Function,
            signature: "fn helper()".to_string(),
            doc: None,
            file: "utils.rs".to_string(),
            line: 5,
        };

        let json = serde_json::to_string(&symbol).expect("Failed to serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse");

        assert_eq!(parsed["doc"], serde_json::Value::Null);
    }

    // =========================================================================
    // IndexStats Tests
    // =========================================================================

    #[test]
    fn index_stats_serializes_to_mcp_contract() {
        let mut languages = HashMap::new();
        languages.insert(Language::Rust, 45);
        languages.insert(Language::TypeScript, 38);
        languages.insert(Language::CSharp, 15);

        let stats = IndexStats {
            files_scanned: 142,
            files_indexed: 98,
            files_skipped: 44,
            symbols_extracted: 567,
            chunks_embedded: 567,
            duration_ms: 3200,
            languages,
        };

        let json = serde_json::to_string(&stats).expect("Failed to serialize IndexStats");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse JSON");

        assert_eq!(parsed["files_scanned"], 142);
        assert_eq!(parsed["files_indexed"], 98);
        assert_eq!(parsed["files_skipped"], 44);
        assert_eq!(parsed["symbols_extracted"], 567);
        assert_eq!(parsed["chunks_embedded"], 567);
        assert_eq!(parsed["duration_ms"], 3200);

        let langs = parsed["languages"]
            .as_object()
            .expect("languages should be an object");
        assert_eq!(langs["rust"], 45);
        assert_eq!(langs["typescript"], 38);
        assert_eq!(langs["csharp"], 15);
    }

    // =========================================================================
    // SearchConfig Tests
    // =========================================================================

    #[test]
    fn search_config_default_is_sensible() {
        let config = SearchConfig::default();

        assert_eq!(config.limit, 10);
        assert_eq!(config.rrf_k, 60);
        assert!(config.use_fts);
        assert!(config.use_vector);
        assert!((config.min_score - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn search_config_round_trips_through_json() {
        let config = SearchConfig {
            limit: 25,
            rrf_k: 100,
            use_fts: true,
            use_vector: false,
            min_score: 0.5,
        };

        let json = serde_json::to_string(&config).expect("Failed to serialize");
        let deserialized: SearchConfig =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized.limit, 25);
        assert_eq!(deserialized.rrf_k, 100);
        assert!(deserialized.use_fts);
        assert!(!deserialized.use_vector);
        assert!((deserialized.min_score - 0.5).abs() < f32::EPSILON);
    }

    // =========================================================================
    // TruesightError Tests
    // =========================================================================

    #[test]
    fn truesight_error_displays_correctly() {
        assert_eq!(
            TruesightError::Parse("unexpected token".to_string()).to_string(),
            "Parse error: unexpected token"
        );
        assert_eq!(
            TruesightError::Database("connection failed".to_string()).to_string(),
            "Database error: connection failed"
        );
        assert_eq!(
            TruesightError::Embedding("model not loaded".to_string()).to_string(),
            "Embedding error: model not loaded"
        );
        assert_eq!(
            TruesightError::UnsupportedLanguage("python".to_string()).to_string(),
            "Unsupported language: python"
        );
    }

    #[test]
    fn truesight_error_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: TruesightError = io_err.into();

        assert!(matches!(err, TruesightError::Io(_)));
        assert!(err.to_string().contains("file not found"));
    }

    // =========================================================================
    // RankedResult Tests
    // =========================================================================

    #[test]
    fn ranked_result_contains_all_scores() {
        let unit = CodeUnit {
            name: "test".to_string(),
            kind: CodeUnitKind::Function,
            signature: "fn test()".to_string(),
            doc: None,
            file_path: PathBuf::from("test.rs"),
            line_start: 1,
            line_end: 5,
            content: "fn test() {}".to_string(),
            parent: None,
            language: Language::Rust,
        };

        let ranked = RankedResult {
            unit,
            fts_score: Some(0.8),
            vector_score: Some(0.9),
            combined_score: 0.85,
            match_type: MatchType::Hybrid,
        };

        let json = serde_json::to_string(&ranked).expect("Failed to serialize");
        let parsed: serde_json::Value = serde_json::from_str(&json).expect("Failed to parse");

        assert_eq!(parsed["fts_score"], 0.8);
        assert_eq!(parsed["vector_score"], 0.9);
        assert_eq!(parsed["combined_score"], 0.85);
        assert_eq!(parsed["match_type"], "hybrid");
        assert_eq!(parsed["unit"]["name"], "test");
    }

    // =========================================================================
    // IndexMetadata Tests
    // =========================================================================

    #[test]
    fn index_metadata_round_trips_through_json() {
        let meta = IndexMetadata {
            repo_id: "/path/to/repo".to_string(),
            branch: "main".to_string(),
            last_indexed_at: chrono::Utc::now(),
            last_commit_sha: Some("abc123".to_string()),
            file_count: 100,
            symbol_count: 500,
        };

        let json = serde_json::to_string(&meta).expect("Failed to serialize");
        let deserialized: IndexMetadata =
            serde_json::from_str(&json).expect("Failed to deserialize");

        assert_eq!(deserialized.repo_id, meta.repo_id);
        assert_eq!(deserialized.branch, meta.branch);
        assert_eq!(deserialized.last_commit_sha, meta.last_commit_sha);
        assert_eq!(deserialized.file_count, meta.file_count);
        assert_eq!(deserialized.symbol_count, meta.symbol_count);
    }
}
