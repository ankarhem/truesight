use std::path::Path;

use tree_sitter::Node;
use truesight_core::{CodeUnit, CodeUnitKind, Language, Result};

use super::helpers::{build_unit, is_csharp_public, parent_name};

pub(super) fn extract_csharp_units(
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
