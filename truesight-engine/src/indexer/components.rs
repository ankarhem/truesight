use std::path::Path;

pub(crate) use crate::parser::CodeParser;
pub(crate) use crate::walker::{DiscoveredFile, FileWalker};
use truesight_core::{CodeUnit, Language, Result};

pub(crate) trait FileDiscovery: Send + Sync {
    fn walk(&self, root: &Path) -> Result<Vec<DiscoveredFile>>;
}

pub(crate) trait SourceParser: Send + Sync {
    fn parse_file(&self, path: &Path, source: &[u8], language: Language) -> Result<Vec<CodeUnit>>;
}

impl FileDiscovery for FileWalker {
    fn walk(&self, root: &Path) -> Result<Vec<DiscoveredFile>> {
        FileWalker::walk(self, root)
    }
}

impl SourceParser for CodeParser {
    fn parse_file(&self, path: &Path, source: &[u8], language: Language) -> Result<Vec<CodeUnit>> {
        CodeParser::parse_file(self, path, source, language)
    }
}
