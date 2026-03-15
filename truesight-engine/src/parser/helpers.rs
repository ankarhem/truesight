use std::path::{Path, PathBuf};

use tree_sitter::{Node, TreeCursor};
use truesight_core::{CodeUnit, CodeUnitKind, Language, Result, TruesightError};

pub(super) fn build_unit(
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

pub(super) fn signature_for_node(node: Node<'_>, source: &[u8]) -> String {
    let text = node.utf8_text(source).unwrap_or_default();
    let boundary = text.find('{').or_else(|| text.find('='));
    let snippet = match boundary {
        Some(index) => &text[..index],
        None => text.trim_end_matches(';'),
    };
    snippet.trim().trim_end_matches(';').trim().to_string()
}

pub(super) fn extract_doc(source: &[u8], start_row: usize) -> Option<String> {
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

pub(super) fn is_rust_public(node: Node<'_>, source: &[u8]) -> bool {
    signature_for_node(node, source).starts_with("pub ")
}

pub(super) fn rust_impl_parent(node: Node<'_>, source: &[u8]) -> Option<String> {
    node.child_by_field_name("type")
        .or_else(|| node.child_by_field_name("trait"))
        .and_then(|child| child.utf8_text(source).ok())
        .map(|text| text.trim().to_string())
}

pub(super) fn has_ts_export(node: Node<'_>, source: &[u8]) -> bool {
    signature_for_node(node, source).starts_with("export ")
}

pub(super) fn is_typescript_public_method(node: Node<'_>, source: &[u8]) -> bool {
    let signature = signature_for_node(node, source);
    !signature.starts_with("private ") && !signature.starts_with("protected ")
}

pub(super) fn is_csharp_public(node: Node<'_>, source: &[u8]) -> bool {
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

pub(super) fn first_named_child(node: Node<'_>) -> Option<Node<'_>> {
    let mut cursor = node.walk();
    let child = node.named_children(&mut cursor).next();
    child
}

pub(super) fn parent_name(parents: &[String]) -> Option<String> {
    parents.last().cloned()
}
