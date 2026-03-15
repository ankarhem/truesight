use std::path::Path;

use tree_sitter::{Language as TsLanguage, Node, Parser};
use truesight_core::{CodeUnit, Language, Result, TruesightError};

mod csharp;
mod helpers;
mod rust;
mod typescript;

type Extractor = fn(Node<'_>, &[u8], &Path, &mut Vec<CodeUnit>, &[String]) -> Result<()>;

struct LanguageConfig {
    language: TsLanguage,
    extractor: Extractor,
}

pub struct CodeParser;

impl CodeParser {
    pub fn new() -> Result<Self> {
        Ok(Self)
    }

    pub fn parse_file(
        &self,
        path: &Path,
        source: &[u8],
        language: Language,
    ) -> Result<Vec<CodeUnit>> {
        let config = language_config(language);
        let mut parser = Parser::new();
        parser.set_language(&config.language).map_err(|error| {
            TruesightError::Parse(format!("failed to set parser language: {error}"))
        })?;

        let tree = parser
            .parse(source, None)
            .ok_or_else(|| TruesightError::Parse(format!("failed to parse {}", path.display())))?;

        let mut units = Vec::new();
        let parents = Vec::new();
        (config.extractor)(tree.root_node(), source, path, &mut units, &parents)?;
        Ok(units)
    }
}

pub fn parse_file(path: &Path, source: &[u8], language: Language) -> Result<Vec<CodeUnit>> {
    CodeParser::new()?.parse_file(path, source, language)
}

fn language_config(language: Language) -> LanguageConfig {
    match language {
        Language::Rust => LanguageConfig {
            language: tree_sitter_rust::LANGUAGE.into(),
            extractor: rust::extract_rust_units,
        },
        Language::TypeScript => LanguageConfig {
            language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            extractor: typescript::extract_typescript_units,
        },
        Language::CSharp => LanguageConfig {
            language: tree_sitter_c_sharp::LANGUAGE.into(),
            extractor: csharp::extract_csharp_units,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fs;
    use std::path::{Path, PathBuf};

    use serde::Deserialize;
    use truesight_core::{CodeUnit, CodeUnitKind, Language};

    use super::parse_file;

    const FIXTURES_ROOT: &str = "../tests/fixtures";

    #[derive(Debug, Deserialize)]
    struct ExpectedFixture {
        symbols: Vec<ExpectedSymbol>,
    }

    #[derive(Debug, Deserialize)]
    struct ExpectedSymbol {
        name: String,
        kind: String,
    }

    #[test]
    fn language_from_path_maps_supported_extensions() {
        assert_eq!(
            Language::from_path(Path::new("src/lib.rs")),
            Some(Language::Rust)
        );
        assert_eq!(
            Language::from_path(Path::new("src/index.ts")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            Language::from_path(Path::new("src/component.tsx")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            Language::from_path(Path::new("Program.cs")),
            Some(Language::CSharp)
        );
        assert_eq!(Language::from_path(Path::new("README.md")), None);
    }

    #[test]
    fn parse_rust_fixture_extracts_expected_symbols() {
        let units = parse_fixture("rust-fixture", &["src/lib.rs", "src/utils.rs"]);

        assert_fixture_symbols("rust-fixture", &units);
        assert!(
            units
                .iter()
                .any(|unit| unit.kind == CodeUnitKind::Struct && unit.name == "User")
        );
        assert!(units.iter().any(
            |unit| unit.kind == CodeUnitKind::Method && unit.parent.as_deref() == Some("User")
        ));
    }

    #[test]
    fn parse_typescript_fixture_extracts_expected_symbols() {
        let units = parse_fixture("ts-fixture", &["src/index.ts", "src/utils.ts"]);

        assert_fixture_symbols("ts-fixture", &units);
        assert!(
            units
                .iter()
                .any(|unit| unit.kind == CodeUnitKind::Class && unit.name == "AuthService")
        );
        assert!(units.iter().any(|unit| unit.kind == CodeUnitKind::Method
            && unit.parent.as_deref() == Some("AuthService")));
    }

    #[test]
    fn parse_csharp_fixture_extracts_expected_symbols() {
        let units = parse_fixture(
            "csharp-fixture",
            &["Program.cs", "Services/AuthService.cs", "Models/User.cs"],
        );

        assert_fixture_symbols("csharp-fixture", &units);
        assert!(
            units
                .iter()
                .any(|unit| unit.kind == CodeUnitKind::Class && unit.name == "Program")
        );
        assert!(
            units
                .iter()
                .any(|unit| unit.name == "Id" && unit.parent.as_deref() == Some("User"))
        );
    }

    #[test]
    fn parser_extracts_docs_signatures_and_ranges() {
        let fixture_root = fixture_root("rust-fixture");
        let source_path = fixture_root.join("src/lib.rs");
        let source = fs::read(&source_path).expect("fixture source should exist");
        let units =
            parse_file(&source_path, &source, Language::Rust).expect("rust fixture should parse");

        let user = find_unit(&units, "User");
        assert_eq!(
            user.doc.as_deref(),
            Some("Represents a user in the system.")
        );
        assert_eq!(user.signature, "pub struct User");
        assert_eq!(user.line_start, 8);
        assert_eq!(user.line_end, 12);

        let new_method = find_unit(&units, "new");
        assert_eq!(new_method.parent.as_deref(), Some("User"));
        assert_eq!(
            new_method.doc.as_deref(),
            Some("Creates a new user with the given details.")
        );
        assert_eq!(
            new_method.signature,
            "pub fn new(id: u64, name: &str, email: &str) -> Self"
        );
        assert!(new_method.content.starts_with("pub fn new("));
    }

    #[test]
    fn parser_handles_syntax_errors_without_panicking() {
        let source = br#"
            /// Healthy function survives surrounding syntax noise.
            pub fn healthy_example(value: &str) -> bool {
                value.contains("x")
            }

            pub fn broken_example(value: &str) -> bool {
                value.contains("y")
        "#;

        let units = parse_file(Path::new("broken.rs"), source, Language::Rust)
            .expect("tree-sitter should tolerate partial syntax errors");

        let healthy = find_unit(&units, "healthy_example");
        assert_eq!(healthy.kind, CodeUnitKind::Function);
        assert_eq!(
            healthy.doc.as_deref(),
            Some("Healthy function survives surrounding syntax noise.")
        );
        assert_eq!(
            healthy.signature,
            "pub fn healthy_example(value: &str) -> bool"
        );
    }

    fn parse_fixture(fixture_name: &str, files: &[&str]) -> Vec<CodeUnit> {
        let mut units = Vec::new();

        for relative_path in files {
            let path = fixture_root(fixture_name).join(relative_path);
            let source = fs::read(&path).expect("fixture source should exist");
            let language =
                Language::from_path(&path).expect("fixture file should map to a language");
            let mut parsed = parse_file(&path, &source, language).expect("fixture should parse");
            units.append(&mut parsed);
        }

        units
    }

    fn assert_fixture_symbols(fixture_name: &str, units: &[CodeUnit]) {
        let expected_path = fixture_root(fixture_name).join("expected.json");
        let expected: ExpectedFixture = serde_json::from_slice(
            &fs::read(expected_path).expect("expected fixture JSON should exist"),
        )
        .expect("expected fixture JSON should parse");

        let actual: HashMap<_, _> = units
            .iter()
            .map(|unit| (unit.name.as_str(), unit.kind))
            .collect();

        for symbol in expected.symbols {
            let expected_kind = expected_kind(&symbol.kind);
            let actual_kind = actual
                .get(symbol.name.as_str())
                .copied()
                .unwrap_or_else(|| panic!("missing symbol {} in {fixture_name}", symbol.name));
            assert_eq!(
                actual_kind, expected_kind,
                "wrong kind for {} in {}",
                symbol.name, fixture_name
            );
        }
    }

    fn expected_kind(kind: &str) -> CodeUnitKind {
        match kind {
            "function" => CodeUnitKind::Function,
            "method" => CodeUnitKind::Method,
            "struct" => CodeUnitKind::Struct,
            "enum" => CodeUnitKind::Enum,
            "trait" => CodeUnitKind::Trait,
            "class" => CodeUnitKind::Class,
            "interface" => CodeUnitKind::Interface,
            "type_alias" | "property" => CodeUnitKind::Constant,
            other => panic!("unsupported fixture kind {other}"),
        }
    }

    fn find_unit<'a>(units: &'a [CodeUnit], name: &str) -> &'a CodeUnit {
        units
            .iter()
            .find(|unit| unit.name == name)
            .unwrap_or_else(|| panic!("missing unit {name}"))
    }

    fn fixture_root(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(FIXTURES_ROOT)
            .join(name)
    }
}
