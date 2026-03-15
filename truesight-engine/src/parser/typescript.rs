use std::path::Path;

use tree_sitter::Node;
use truesight_core::{CodeUnit, CodeUnitKind, Language, Result};

use super::helpers::{
    build_unit, first_named_child, has_ts_export, is_typescript_public_method, parent_name,
};

pub(super) fn extract_typescript_units(
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
