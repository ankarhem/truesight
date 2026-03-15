use std::collections::BTreeMap;
use std::path::Path;

use truesight_core::{RepoMap, Result, Storage, SymbolInfo};

use crate::repo_context::detect_repo_context;

mod dependencies;
mod module_accumulator;
mod paths;

use dependencies::{build_symbol_locations, dependency_hints};
use module_accumulator::ModuleAccumulator;
use paths::{
    file_name_for_symbol, matches_filter, module_name, module_path, normalized_filter,
    relative_path,
};

pub struct RepoMapGenerator;

impl RepoMapGenerator {
    pub async fn generate_for_repo(root: &Path, storage: &dyn Storage) -> Result<RepoMap> {
        let context = detect_repo_context(root)?;
        Self::generate(root, storage, &context.repo_id, &context.branch, None).await
    }

    pub async fn generate(
        root: &Path,
        storage: &dyn Storage,
        repo_id: &str,
        branch: &str,
        filter: Option<&str>,
    ) -> Result<RepoMap> {
        let filter = normalized_filter(filter);
        let units = storage
            .get_all_symbols(repo_id, branch)
            .await?
            .into_iter()
            .filter(|unit| matches_filter(root, &unit.file_path, filter.as_deref()))
            .collect::<Vec<_>>();
        let repo_root = root.to_path_buf();

        if units.is_empty() {
            return Ok(RepoMap {
                repo_root,
                branch: branch.to_string(),
                modules: Vec::new(),
            });
        }

        let symbol_locations = build_symbol_locations(root, &units);
        let mut modules = BTreeMap::<std::path::PathBuf, ModuleAccumulator>::new();

        for unit in units {
            let relative_path = relative_path(root, &unit.file_path);
            let current_module_path = module_path(&relative_path);
            let file_name = file_name_for_symbol(&relative_path);
            let symbol = SymbolInfo {
                name: unit.name.clone(),
                kind: unit.kind,
                signature: unit.signature.clone(),
                doc: unit.doc.clone(),
                file: file_name.clone(),
                line: unit.line_start,
            };
            let module = modules
                .entry(current_module_path.clone())
                .or_insert_with(|| {
                    ModuleAccumulator::new(
                        module_name(root, &current_module_path),
                        current_module_path,
                    )
                });

            module.files.insert(file_name.clone());
            module.units.push(unit);
            module.push_symbol(relative_path, symbol);
        }

        for module in modules.values_mut() {
            module.depends_on =
                dependency_hints(root, &module.path, &module.units, &symbol_locations);
        }

        Ok(RepoMap {
            repo_root,
            branch: branch.to_string(),
            modules: modules
                .into_values()
                .map(ModuleAccumulator::finish)
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use async_trait::async_trait;
    use truesight_core::{
        CodeUnit, CodeUnitKind, IndexMetadata, Language, RankedResult, Storage, TruesightError,
    };

    use super::RepoMapGenerator;

    const REPO_ID: &str = "repo";
    const BRANCH: &str = "main";

    #[tokio::test]
    async fn test_generate_rust_repomap() {
        let root = fixture_root("rust-fixture");
        let storage = StaticStorage::new(rust_fixture_units(&root));

        let repo_map = RepoMapGenerator::generate(&root, &storage, REPO_ID, BRANCH, None)
            .await
            .expect("repo map should generate");

        assert_eq!(repo_map.repo_root, root);
        assert_eq!(repo_map.branch, BRANCH);
        assert_eq!(repo_map.modules.len(), 1);

        let module = &repo_map.modules[0];
        assert_eq!(module.name, "src");
        assert_eq!(module.path, PathBuf::from("src"));
        assert_eq!(module.files, vec!["lib.rs", "utils.rs"]);

        let names: Vec<_> = module
            .symbols
            .iter()
            .map(|symbol| symbol.name.as_str())
            .collect();
        assert_eq!(
            names,
            vec![
                "User",
                "ValidationError",
                "Validatable",
                "new",
                "display_name",
                "validate",
                "validate_email",
                "calculate_checksum",
                "format_name",
                "truncate_text",
                "is_blank",
            ]
        );

        let user = module
            .symbols
            .iter()
            .find(|symbol| symbol.name == "User")
            .expect("User symbol should exist");
        assert_eq!(user.file, "lib.rs");
        assert_eq!(user.line, 8);
        assert_eq!(user.signature, "pub struct User");
        assert_eq!(
            user.doc.as_deref(),
            Some("Represents a user in the system.")
        );

        let format_name = module
            .symbols
            .iter()
            .find(|symbol| symbol.name == "format_name")
            .expect("format_name symbol should exist");
        assert_eq!(format_name.file, "utils.rs");
        assert_eq!(format_name.line, 4);
        assert_eq!(
            format_name.doc.as_deref(),
            Some("Formats a name to title case.")
        );
    }

    #[tokio::test]
    async fn test_repromap_sorts_modules_and_keeps_missing_docs() {
        let root = PathBuf::from("/repo");
        let storage = StaticStorage::new(vec![
            unit(
                root.join("zeta/service.rs"),
                "ZetaService",
                CodeUnitKind::Struct,
                "pub struct ZetaService",
                None,
                4,
                "pub struct ZetaService",
            ),
            unit(
                root.join("alpha/model.rs"),
                "AlphaModel",
                CodeUnitKind::Struct,
                "pub struct AlphaModel",
                Some("Shared model."),
                2,
                "pub struct AlphaModel",
            ),
        ]);

        let repo_map = RepoMapGenerator::generate(&root, &storage, REPO_ID, BRANCH, None)
            .await
            .expect("repo map should generate");

        let module_paths: Vec<_> = repo_map
            .modules
            .iter()
            .map(|module| module.path.clone())
            .collect();
        assert_eq!(
            module_paths,
            vec![PathBuf::from("alpha"), PathBuf::from("zeta")]
        );
        assert_eq!(
            repo_map.modules[0].symbols[0].doc.as_deref(),
            Some("Shared model.")
        );
        assert_eq!(repo_map.modules[1].symbols[0].doc, None);
    }

    #[tokio::test]
    async fn test_repromap_returns_empty_modules_for_empty_repo() {
        let root = PathBuf::from("/empty-repo");
        let storage = StaticStorage::new(Vec::new());

        let repo_map = RepoMapGenerator::generate(&root, &storage, REPO_ID, BRANCH, None)
            .await
            .expect("repo map should generate");

        assert_eq!(repo_map.repo_root, root);
        assert!(repo_map.modules.is_empty());
    }

    #[tokio::test]
    async fn test_repromap_builds_dependency_hints_from_stored_symbols() {
        let root = PathBuf::from("/repo");
        let storage = StaticStorage::new(vec![
            unit(
                root.join("shared/user.rs"),
                "User",
                CodeUnitKind::Struct,
                "pub struct User",
                Some("Shared user."),
                1,
                "pub struct User",
            ),
            unit(
                root.join("services/create.rs"),
                "create_user",
                CodeUnitKind::Function,
                "pub fn create_user(user: User) -> User",
                Some("Creates a user."),
                5,
                "pub fn create_user(user: User) -> User { User }",
            ),
        ]);

        let repo_map = RepoMapGenerator::generate(&root, &storage, REPO_ID, BRANCH, None)
            .await
            .expect("repo map should generate");

        let services = repo_map
            .modules
            .iter()
            .find(|module| module.path == Path::new("services"))
            .expect("services module should exist");
        assert_eq!(services.depends_on, vec![String::from("shared")]);
    }

    #[tokio::test]
    async fn test_repromap_filters_to_matching_path_prefix() {
        let root = fixture_root("rust-fixture");
        let storage = StaticStorage::new(rust_fixture_units(&root));

        let repo_map =
            RepoMapGenerator::generate(&root, &storage, REPO_ID, BRANCH, Some("src/lib.rs"))
                .await
                .expect("filtered repo map should generate");

        assert_eq!(repo_map.modules.len(), 1);
        let module = &repo_map.modules[0];
        assert_eq!(module.path, PathBuf::from("src"));
        assert_eq!(module.files, vec!["lib.rs"]);
        assert!(module.symbols.iter().all(|symbol| symbol.file == "lib.rs"));
    }

    fn rust_fixture_units(root: &Path) -> Vec<CodeUnit> {
        vec![
            unit(
                root.join("src/lib.rs"),
                "User",
                CodeUnitKind::Struct,
                "pub struct User",
                Some("Represents a user in the system."),
                8,
                "pub struct User",
            ),
            unit(
                root.join("src/lib.rs"),
                "ValidationError",
                CodeUnitKind::Enum,
                "pub enum ValidationError",
                Some("Possible errors when validating users."),
                15,
                "pub enum ValidationError",
            ),
            unit(
                root.join("src/lib.rs"),
                "Validatable",
                CodeUnitKind::Trait,
                "pub trait Validatable",
                Some("Trait for entities that can be validated."),
                22,
                "pub trait Validatable",
            ),
            unit(
                root.join("src/lib.rs"),
                "new",
                CodeUnitKind::Method,
                "pub fn new(id: u64, name: &str, email: &str) -> Self",
                Some("Creates a new user with the given details."),
                28,
                "pub fn new(id: u64, name: &str, email: &str) -> Self { format_name(name) }",
            ),
            unit(
                root.join("src/lib.rs"),
                "display_name",
                CodeUnitKind::Method,
                "pub fn display_name(&self) -> String",
                Some("Returns a formatted display string for the user."),
                37,
                "pub fn display_name(&self) -> String",
            ),
            unit(
                root.join("src/lib.rs"),
                "validate",
                CodeUnitKind::Method,
                "fn validate(&self) -> Result<(), ValidationError>",
                Some("Trait for entities that can be validated."),
                43,
                "fn validate(&self) -> Result<(), ValidationError>",
            ),
            unit(
                root.join("src/lib.rs"),
                "validate_email",
                CodeUnitKind::Function,
                "pub fn validate_email(email: &str) -> bool",
                Some("Validates an email address format."),
                58,
                "pub fn validate_email(email: &str) -> bool",
            ),
            unit(
                root.join("src/lib.rs"),
                "calculate_checksum",
                CodeUnitKind::Function,
                "pub fn calculate_checksum(data: &[u8]) -> u32",
                Some("Calculates a simple checksum for data integrity."),
                63,
                "pub fn calculate_checksum(data: &[u8]) -> u32",
            ),
            unit(
                root.join("src/utils.rs"),
                "format_name",
                CodeUnitKind::Function,
                "pub fn format_name(name: &str) -> String",
                Some("Formats a name to title case."),
                4,
                "pub fn format_name(name: &str) -> String",
            ),
            unit(
                root.join("src/utils.rs"),
                "truncate_text",
                CodeUnitKind::Function,
                "pub fn truncate_text(text: &str, max_len: usize) -> String",
                Some("Truncates text to a maximum length with ellipsis."),
                22,
                "pub fn truncate_text(text: &str, max_len: usize) -> String",
            ),
            unit(
                root.join("src/utils.rs"),
                "is_blank",
                CodeUnitKind::Function,
                "pub fn is_blank(s: &str) -> bool",
                Some("Checks if a string is blank (empty or whitespace only)."),
                31,
                "pub fn is_blank(s: &str) -> bool",
            ),
        ]
    }

    fn unit(
        file_path: PathBuf,
        name: &str,
        kind: CodeUnitKind,
        signature: &str,
        doc: Option<&str>,
        line_start: u32,
        content: &str,
    ) -> CodeUnit {
        CodeUnit {
            name: name.to_string(),
            kind,
            signature: signature.to_string(),
            doc: doc.map(str::to_string),
            file_path,
            line_start,
            line_end: line_start,
            content: content.to_string(),
            parent: None,
            language: Language::Rust,
        }
    }

    fn fixture_root(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/fixtures")
            .join(name)
    }

    struct StaticStorage {
        units: Vec<CodeUnit>,
    }

    impl StaticStorage {
        fn new(units: Vec<CodeUnit>) -> Self {
            Self { units }
        }
    }

    #[async_trait]
    impl Storage for StaticStorage {
        async fn store_code_units(
            &self,
            _repo_id: &str,
            _branch: &str,
            _units: &[CodeUnit],
        ) -> truesight_core::Result<()> {
            Err(TruesightError::Index(String::from(
                "store_code_units should not be called in repomap tests",
            )))
        }

        async fn search_fts(
            &self,
            _repo_id: &str,
            _branch: &str,
            _query: &str,
            _limit: usize,
        ) -> truesight_core::Result<Vec<RankedResult>> {
            Err(TruesightError::Index(String::from(
                "search_fts should not be called in repomap tests",
            )))
        }

        async fn search_vector(
            &self,
            _repo_id: &str,
            _branch: &str,
            _embedding: &[f32],
            _limit: usize,
        ) -> truesight_core::Result<Vec<RankedResult>> {
            Err(TruesightError::Index(String::from(
                "search_vector should not be called in repomap tests",
            )))
        }

        async fn search_hybrid(
            &self,
            _repo_id: &str,
            _branch: &str,
            _query: &str,
            _embedding: &[f32],
            _limit: usize,
            _rrf_k: u32,
        ) -> truesight_core::Result<Vec<RankedResult>> {
            Err(TruesightError::Index(String::from(
                "search_hybrid should not be called in repomap tests",
            )))
        }

        async fn get_index_metadata(
            &self,
            _repo_id: &str,
            _branch: &str,
        ) -> truesight_core::Result<Option<IndexMetadata>> {
            Err(TruesightError::Index(String::from(
                "get_index_metadata should not be called in repomap tests",
            )))
        }

        async fn set_index_metadata(
            &self,
            _repo_id: &str,
            _branch: &str,
            _meta: &IndexMetadata,
        ) -> truesight_core::Result<()> {
            Err(TruesightError::Index(String::from(
                "set_index_metadata should not be called in repomap tests",
            )))
        }

        async fn has_indexed_symbols(
            &self,
            _repo_id: &str,
            _branch: &str,
        ) -> truesight_core::Result<bool> {
            Err(TruesightError::Index(String::from(
                "has_indexed_symbols should not be called in repomap tests",
            )))
        }

        async fn delete_branch_index(
            &self,
            _repo_id: &str,
            _branch: &str,
        ) -> truesight_core::Result<()> {
            Err(TruesightError::Index(String::from(
                "delete_branch_index should not be called in repomap tests",
            )))
        }

        async fn get_all_symbols(
            &self,
            _repo_id: &str,
            _branch: &str,
        ) -> truesight_core::Result<Vec<CodeUnit>> {
            Ok(self.units.clone())
        }
    }
}
