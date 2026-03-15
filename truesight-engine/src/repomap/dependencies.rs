use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use truesight_core::CodeUnit;

use super::paths::{module_label, module_path, relative_path};

pub(super) fn build_symbol_locations(
    root: &Path,
    units: &[CodeUnit],
) -> BTreeMap<String, BTreeSet<String>> {
    let mut locations = BTreeMap::<String, BTreeSet<String>>::new();

    for unit in units {
        let relative_path = relative_path(root, &unit.file_path);
        let label = module_label(root, &module_path(&relative_path));
        locations
            .entry(unit.name.clone())
            .or_default()
            .insert(label);
    }

    locations
}

pub(super) fn dependency_hints(
    root: &Path,
    current_module_path: &Path,
    units: &[CodeUnit],
    symbol_locations: &BTreeMap<String, BTreeSet<String>>,
) -> Vec<String> {
    let current_label = module_label(root, current_module_path);
    let mut depends_on = BTreeSet::new();

    for unit in units {
        for token in identifier_tokens(&unit.signature)
            .into_iter()
            .chain(identifier_tokens(&unit.content))
        {
            if let Some(modules) = symbol_locations.get(&token) {
                for module in modules {
                    if module != &current_label {
                        depends_on.insert(module.clone());
                    }
                }
            }
        }
    }

    depends_on.into_iter().collect()
}

fn identifier_tokens(text: &str) -> Vec<String> {
    let mut tokens = BTreeSet::new();
    let mut current = String::new();

    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            current.push(ch);
        } else if !current.is_empty() {
            tokens.insert(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        tokens.insert(current);
    }

    tokens.into_iter().collect()
}
