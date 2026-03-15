use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::CodeUnitKind;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SymbolInfo {
    pub name: String,
    pub kind: CodeUnitKind,
    pub signature: String,
    pub doc: Option<String>,
    pub file: String,
    pub line: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ModuleInfo {
    pub name: String,
    pub path: PathBuf,
    pub files: Vec<String>,
    pub symbols: Vec<SymbolInfo>,
    pub depends_on: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RepoMap {
    pub repo_root: PathBuf,
    pub branch: String,
    pub modules: Vec<ModuleInfo>,
}
