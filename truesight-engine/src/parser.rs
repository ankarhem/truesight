use std::path::{Path, PathBuf};

use tree_sitter::{Language as TsLanguage, Node, Parser, TreeCursor};
use truesight_core::{CodeUnit, CodeUnitKind, Language, Result, TruesightError};

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
            extractor: extract_rust_units,
        },
        Language::TypeScript => LanguageConfig {
            language: tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            extractor: extract_typescript_units,
        },
        Language::CSharp => LanguageConfig {
            language: tree_sitter_c_sharp::LANGUAGE.into(),
            extractor: extract_csharp_units,
        },
    }
}

fn extract_rust_units(
    node: Node<'_>,
    source: &[u8],
    path: &Path,
    units: &mut Vec<CodeUnit>,
    parents: &[String],
) -> Result<()> {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "struct_item" if is_rust_public(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::Rust,
                    CodeUnitKind::Struct,
                    parent_name(parents),
                )? {
                    units.push(unit);
                }
            }
            "enum_item" if is_rust_public(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::Rust,
                    CodeUnitKind::Enum,
                    parent_name(parents),
                )? {
                    units.push(unit);
                }
            }
            "trait_item" if is_rust_public(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::Rust,
                    CodeUnitKind::Trait,
                    parent_name(parents),
                )? {
                    let trait_name = unit.name.clone();
                    units.push(unit);
                    let mut next_parents = parents.to_vec();
                    next_parents.push(trait_name);
                    extract_rust_trait_methods(child, source, path, units, &next_parents)?;
                }
            }
            "function_item" if is_rust_public(child, source) && parents.is_empty() => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::Rust,
                    CodeUnitKind::Function,
                    None,
                )? {
                    units.push(unit);
                }
            }
            "impl_item" => {
                let mut next_parents = parents.to_vec();
                if let Some(parent) = rust_impl_parent(child, source) {
                    next_parents.push(parent);
                }
                extract_rust_impl_methods(child, source, path, units, &next_parents)?;
            }
            _ => extract_rust_units(child, source, path, units, parents)?,
        }
    }
    Ok(())
}

fn extract_rust_trait_methods(
    node: Node<'_>,
    source: &[u8],
    path: &Path,
    units: &mut Vec<CodeUnit>,
    parents: &[String],
) -> Result<()> {
    let body = match node.child_by_field_name("body") {
        Some(body) => body,
        None => return Ok(()),
    };
    let mut cursor = body.walk();
    for child in body.named_children(&mut cursor) {
        if child.kind() == "function_signature_item" {
            if let Some(unit) = build_unit(
                child,
                source,
                path,
                Language::Rust,
                CodeUnitKind::Method,
                parent_name(parents),
            )? {
                units.push(unit);
            }
        }
    }
    Ok(())
}

fn extract_rust_impl_methods(
    node: Node<'_>,
    source: &[u8],
    path: &Path,
    units: &mut Vec<CodeUnit>,
    parents: &[String],
) -> Result<()> {
    let body = match node.child_by_field_name("body") {
        Some(body) => body,
        None => return Ok(()),
    };
    let mut cursor = body.walk();
    for child in body.named_children(&mut cursor) {
        if child.kind() == "function_item" {
            if let Some(unit) = build_unit(
                child,
                source,
                path,
                Language::Rust,
                CodeUnitKind::Method,
                parent_name(parents),
            )? {
                units.push(unit);
            }
        }
    }
    Ok(())
}

fn extract_typescript_units(
    node: Node<'_>,
    source: &[u8],
    path: &Path,
    units: &mut Vec<CodeUnit>,
    parents: &[String],
) -> Result<()> {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "export_statement" => extract_typescript_export(child, source, path, units, parents)?,
            "class_declaration" if parents.is_empty() && has_ts_export(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::TypeScript,
                    CodeUnitKind::Class,
                    None,
                )? {
                    let class_name = unit.name.clone();
                    units.push(unit);
                    let mut next_parents = parents.to_vec();
                    next_parents.push(class_name);
                    extract_typescript_class_members(child, source, path, units, &next_parents)?;
                }
            }
            _ => extract_typescript_units(child, source, path, units, parents)?,
        }
    }
    Ok(())
}

fn extract_typescript_export(
    node: Node<'_>,
    source: &[u8],
    path: &Path,
    units: &mut Vec<CodeUnit>,
    parents: &[String],
) -> Result<()> {
    let Some(declaration) = first_named_child(node) else {
        return Ok(());
    };

    match declaration.kind() {
        "interface_declaration" => {
            if let Some(unit) = build_unit(
                declaration,
                source,
                path,
                Language::TypeScript,
                CodeUnitKind::Interface,
                None,
            )? {
                units.push(unit);
            }
        }
        "type_alias_declaration" => {
            if let Some(unit) = build_unit(
                declaration,
                source,
                path,
                Language::TypeScript,
                CodeUnitKind::Constant,
                None,
            )? {
                units.push(unit);
            }
        }
        "function_declaration" => {
            if let Some(unit) = build_unit(
                declaration,
                source,
                path,
                Language::TypeScript,
                CodeUnitKind::Function,
                None,
            )? {
                units.push(unit);
            }
        }
        "class_declaration" => {
            if let Some(unit) = build_unit(
                declaration,
                source,
                path,
                Language::TypeScript,
                CodeUnitKind::Class,
                None,
            )? {
                let class_name = unit.name.clone();
                units.push(unit);
                let mut next_parents = parents.to_vec();
                next_parents.push(class_name);
                extract_typescript_class_members(declaration, source, path, units, &next_parents)?;
            }
        }
        _ => {}
    }

    Ok(())
}

fn extract_typescript_class_members(
    node: Node<'_>,
    source: &[u8],
    path: &Path,
    units: &mut Vec<CodeUnit>,
    parents: &[String],
) -> Result<()> {
    let Some(body) = node.child_by_field_name("body") else {
        return Ok(());
    };

    let mut cursor = body.walk();
    for child in body.named_children(&mut cursor) {
        if child.kind() == "method_definition" && is_typescript_public_method(child, source) {
            if let Some(unit) = build_unit(
                child,
                source,
                path,
                Language::TypeScript,
                CodeUnitKind::Method,
                parent_name(parents),
            )? {
                units.push(unit);
            }
        }
    }

    Ok(())
}

fn extract_csharp_units(
    node: Node<'_>,
    source: &[u8],
    path: &Path,
    units: &mut Vec<CodeUnit>,
    parents: &[String],
) -> Result<()> {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "class_declaration" if is_csharp_public(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::CSharp,
                    CodeUnitKind::Class,
                    parent_name(parents),
                )? {
                    let class_name = unit.name.clone();
                    units.push(unit);
                    let mut next_parents = parents.to_vec();
                    next_parents.push(class_name);
                    extract_csharp_class_members(child, source, path, units, &next_parents)?;
                }
            }
            "interface_declaration" if is_csharp_public(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::CSharp,
                    CodeUnitKind::Interface,
                    parent_name(parents),
                )? {
                    units.push(unit);
                }
            }
            "enum_declaration" if is_csharp_public(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::CSharp,
                    CodeUnitKind::Enum,
                    parent_name(parents),
                )? {
                    units.push(unit);
                }
            }
            _ => extract_csharp_units(child, source, path, units, parents)?,
        }
    }
    Ok(())
}

fn extract_csharp_class_members(
    node: Node<'_>,
    source: &[u8],
    path: &Path,
    units: &mut Vec<CodeUnit>,
    parents: &[String],
) -> Result<()> {
    let Some(body) = node.child_by_field_name("body") else {
        return Ok(());
    };

    let mut cursor = body.walk();
    for child in body.named_children(&mut cursor) {
        match child.kind() {
            "method_declaration" if is_csharp_public(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::CSharp,
                    CodeUnitKind::Method,
                    parent_name(parents),
                )? {
                    units.push(unit);
                }
            }
            "property_declaration" if is_csharp_public(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::CSharp,
                    CodeUnitKind::Constant,
                    parent_name(parents),
                )? {
                    units.push(unit);
                }
            }
            "enum_declaration" if is_csharp_public(child, source) => {
                if let Some(unit) = build_unit(
                    child,
                    source,
                    path,
                    Language::CSharp,
                    CodeUnitKind::Enum,
                    parent_name(parents),
                )? {
                    units.push(unit);
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn build_unit(
    node: Node<'_>,
    source: &[u8],
    path: &Path,
    language: Language,
    kind: CodeUnitKind,
    parent: Option<String>,
) -> Result<Option<CodeUnit>> {
    let Some(name_node) = node
        .child_by_field_name("name")
        .or_else(|| named_identifier_child(node))
    else {
        return Ok(None);
    };

    let name = node_text(name_node, source)?;
    let content = node_text(node, source)?;
    let signature = signature_for_node(node, source);
    let doc = extract_doc(source, node.start_position().row);

    Ok(Some(CodeUnit {
        name,
        kind,
        signature,
        doc,
        file_path: PathBuf::from(path),
        line_start: node.start_position().row as u32 + 1,
        line_end: node.end_position().row as u32 + 1,
        content,
        parent,
        language,
    }))
}

fn signature_for_node(node: Node<'_>, source: &[u8]) -> String {
    let text = node.utf8_text(source).unwrap_or_default();
    let boundary = text.find('{').or_else(|| text.find('='));
    let snippet = match boundary {
        Some(index) => &text[..index],
        None => text.trim_end_matches(';'),
    };
    snippet.trim().trim_end_matches(';').trim().to_string()
}

fn extract_doc(source: &[u8], start_row: usize) -> Option<String> {
    let text = std::str::from_utf8(source).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    if start_row == 0 || start_row > lines.len() {
        return None;
    }

    let mut docs = Vec::new();
    let mut reverse_docs = true;
    let mut row = start_row;

    while row > 0 {
        row -= 1;
        let trimmed = lines[row].trim();
        if trimmed.is_empty() {
            break;
        }

        if let Some(doc_line) = strip_line_doc(trimmed) {
            docs.push(doc_line);
            continue;
        }

        if trimmed.starts_with("*/") {
            let mut block = vec![strip_block_doc(trimmed)];
            while row > 0 {
                row -= 1;
                let block_line = lines[row].trim();
                block.push(strip_block_doc(block_line));
                if block_line.starts_with("/**") || block_line.starts_with("/*") {
                    break;
                }
            }
            block.reverse();
            docs = block;
            reverse_docs = false;
            break;
        }

        break;
    }

    if reverse_docs {
        docs.reverse();
    }
    let joined = docs
        .into_iter()
        .map(|line| sanitize_doc_line(&line))
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    (!joined.is_empty()).then_some(joined)
}

fn strip_line_doc(line: &str) -> Option<String> {
    if let Some(rest) = line.strip_prefix("///") {
        return Some(rest.trim().to_string());
    }
    if let Some(rest) = line.strip_prefix("//!") {
        return Some(rest.trim().to_string());
    }
    None
}

fn strip_block_doc(line: &str) -> String {
    line.trim()
        .trim_start_matches("/**")
        .trim_start_matches("/*")
        .trim_start_matches('*')
        .trim_start_matches("///")
        .trim_end_matches("*/")
        .trim()
        .to_string()
}

fn sanitize_doc_line(line: &str) -> String {
    line.replace("<summary>", "")
        .replace("</summary>", "")
        .trim()
        .to_string()
}

fn is_rust_public(node: Node<'_>, source: &[u8]) -> bool {
    signature_for_node(node, source).starts_with("pub ")
}

fn rust_impl_parent(node: Node<'_>, source: &[u8]) -> Option<String> {
    node.child_by_field_name("type")
        .or_else(|| node.child_by_field_name("trait"))
        .and_then(|child| child.utf8_text(source).ok())
        .map(|text| text.trim().to_string())
}

fn has_ts_export(node: Node<'_>, source: &[u8]) -> bool {
    signature_for_node(node, source).starts_with("export ")
}

fn is_typescript_public_method(node: Node<'_>, source: &[u8]) -> bool {
    let signature = signature_for_node(node, source);
    !signature.starts_with("private ") && !signature.starts_with("protected ")
}

fn is_csharp_public(node: Node<'_>, source: &[u8]) -> bool {
    signature_for_node(node, source).contains("public ")
}

fn node_text(node: Node<'_>, source: &[u8]) -> Result<String> {
    node.utf8_text(source)
        .map(|text| text.to_string())
        .map_err(|error| TruesightError::Parse(format!("invalid UTF-8 in source: {error}")))
}

fn named_identifier_child(node: Node<'_>) -> Option<Node<'_>> {
    let mut cursor: TreeCursor<'_> = node.walk();
    let found = node.named_children(&mut cursor).find(|child| {
        matches!(
            child.kind(),
            "identifier" | "type_identifier" | "property_identifier"
        )
    });
    found
}

fn first_named_child(node: Node<'_>) -> Option<Node<'_>> {
    let mut cursor = node.walk();
    let child = node.named_children(&mut cursor).next();
    child
}

fn parent_name(parents: &[String]) -> Option<String> {
    parents.last().cloned()
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
