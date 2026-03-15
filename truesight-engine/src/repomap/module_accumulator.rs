use std::collections::BTreeSet;
use std::path::PathBuf;

use truesight_core::{CodeUnit, ModuleInfo, SymbolInfo};

pub(super) struct ModuleAccumulator {
    pub(super) name: String,
    pub(super) path: PathBuf,
    pub(super) files: BTreeSet<String>,
    pub(super) units: Vec<CodeUnit>,
    symbols: Vec<OrderedSymbol>,
    pub(super) depends_on: Vec<String>,
}

impl ModuleAccumulator {
    pub(super) fn new(name: String, path: PathBuf) -> Self {
        Self {
            name,
            path,
            files: BTreeSet::new(),
            units: Vec::new(),
            symbols: Vec::new(),
            depends_on: Vec::new(),
        }
    }

    pub(super) fn push_symbol(&mut self, relative_path: PathBuf, symbol: SymbolInfo) {
        self.symbols.push(OrderedSymbol {
            file_sort_key: relative_path,
            line: symbol.line,
            name_sort_key: symbol.name.clone(),
            symbol,
        });
    }

    pub(super) fn finish(mut self) -> ModuleInfo {
        self.symbols.sort_by(|left, right| {
            left.file_sort_key
                .cmp(&right.file_sort_key)
                .then_with(|| left.line.cmp(&right.line))
                .then_with(|| left.name_sort_key.cmp(&right.name_sort_key))
        });

        ModuleInfo {
            name: self.name,
            path: self.path,
            files: self.files.into_iter().collect(),
            symbols: self.symbols.into_iter().map(|entry| entry.symbol).collect(),
            depends_on: self.depends_on,
        }
    }
}

struct OrderedSymbol {
    file_sort_key: PathBuf,
    line: u32,
    name_sort_key: String,
    symbol: SymbolInfo,
}
