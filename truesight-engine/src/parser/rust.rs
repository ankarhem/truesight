use std::path::Path;

use tree_sitter::Node;
use truesight_core::{CodeUnit, CodeUnitKind, Language, Result};

use super::helpers::{build_unit, is_rust_public, parent_name, rust_impl_parent};

pub(super) fn extract_rust_units(
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
