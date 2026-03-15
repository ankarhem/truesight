use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    Rust,
    TypeScript,
    CSharp,
}

impl Language {
    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["rs"],
            Language::TypeScript => &["ts", "tsx"],
            Language::CSharp => &["cs"],
        }
    }

    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some("rs") => Some(Language::Rust),
            Some("ts") | Some("tsx") => Some(Language::TypeScript),
            Some("cs") => Some(Language::CSharp),
            _ => None,
        }
    }
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::Rust => write!(f, "rust"),
            Language::TypeScript => write!(f, "typescript"),
            Language::CSharp => write!(f, "csharp"),
        }
    }
}

impl std::str::FromStr for Language {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "rust" => Ok(Language::Rust),
            "typescript" => Ok(Language::TypeScript),
            "csharp" => Ok(Language::CSharp),
            other => Err(format!("unknown language: {other}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum CodeUnitKind {
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    Class,
    Interface,
    Constant,
    Module,
}

impl std::fmt::Display for CodeUnitKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CodeUnitKind::Function => write!(f, "function"),
            CodeUnitKind::Method => write!(f, "method"),
            CodeUnitKind::Struct => write!(f, "struct"),
            CodeUnitKind::Enum => write!(f, "enum"),
            CodeUnitKind::Trait => write!(f, "trait"),
            CodeUnitKind::Class => write!(f, "class"),
            CodeUnitKind::Interface => write!(f, "interface"),
            CodeUnitKind::Constant => write!(f, "constant"),
            CodeUnitKind::Module => write!(f, "module"),
        }
    }
}

impl std::str::FromStr for CodeUnitKind {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "function" => Ok(CodeUnitKind::Function),
            "method" => Ok(CodeUnitKind::Method),
            "struct" => Ok(CodeUnitKind::Struct),
            "enum" => Ok(CodeUnitKind::Enum),
            "trait" => Ok(CodeUnitKind::Trait),
            "class" => Ok(CodeUnitKind::Class),
            "interface" => Ok(CodeUnitKind::Interface),
            "constant" => Ok(CodeUnitKind::Constant),
            "module" => Ok(CodeUnitKind::Module),
            other => Err(format!("unknown code unit kind: {other}")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CodeUnit {
    pub name: String,
    pub kind: CodeUnitKind,
    pub signature: String,
    pub doc: Option<String>,
    pub file_path: PathBuf,
    pub line_start: u32,
    pub line_end: u32,
    pub content: String,
    pub parent: Option<String>,
    pub language: Language,
}
