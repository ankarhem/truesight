use std::fs;
use std::path::{Path, PathBuf};

use truesight_core::{IndexedCodeUnit, Language, Result};

use super::components::SourceParser;
use crate::indexing::{build_pending_units, repo_relative_path};
use crate::util::hash_bytes;
use crate::walker::DiscoveredFile;

#[derive(Debug, Clone)]
pub(super) struct ProcessedFile {
    pub(super) path: PathBuf,
    pub(super) language: Language,
    pub(super) file_hash: String,
    pub(super) units: Vec<IndexedCodeUnit>,
}

pub(super) fn process_file<P>(
    index: usize,
    total: usize,
    root: &Path,
    file: &DiscoveredFile,
    parser: &P,
) -> Result<ProcessedFile>
where
    P: SourceParser,
{
    tracing::info!(file_number = index, total_files = total, path = %file.path.display(), "indexing file");

    let source = fs::read(&file.path)?;
    let file_hash = hash_bytes(&source);
    let relative_path = repo_relative_path(root, &file.path)?;
    let parsed_units = parser.parse_file(&relative_path, &source, file.language)?;

    if parsed_units.is_empty() {
        return Ok(ProcessedFile {
            path: relative_path,
            language: file.language,
            file_hash,
            units: Vec::new(),
        });
    }

    let units = build_pending_units(parsed_units, &file_hash);

    Ok(ProcessedFile {
        path: relative_path,
        language: file.language,
        file_hash,
        units,
    })
}
