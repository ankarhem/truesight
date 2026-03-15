use std::io::Write;
use std::path::Path;

use anyhow::Result;
use truesight_core::RepoMap;

use super::{IndexRepoResponse, SearchRepoResponse};

pub(super) fn write_index_output(
    writer: &mut dyn Write,
    response: &IndexRepoResponse,
    full: bool,
) -> Result<()> {
    writeln!(writer, "repo_root: {}", response.repo_root.display())?;
    writeln!(writer, "branch: {}", response.branch)?;
    writeln!(
        writer,
        "mode: {}",
        if full { "full" } else { "incremental" }
    )?;
    writeln!(writer, "files_scanned: {}", response.stats.files_scanned)?;
    writeln!(writer, "files_indexed: {}", response.stats.files_indexed)?;
    writeln!(writer, "files_skipped: {}", response.stats.files_skipped)?;
    writeln!(
        writer,
        "symbols_extracted: {}",
        response.stats.symbols_extracted
    )?;
    writeln!(
        writer,
        "chunks_embedded: {}",
        response.stats.chunks_embedded
    )?;
    writeln!(writer, "duration_ms: {}", response.stats.duration_ms)?;
    writeln!(writer, "languages:")?;

    let mut items = response.languages.iter().collect::<Vec<_>>();
    items.sort_by(|left, right| left.0.cmp(right.0));
    for (language, count) in items {
        writeln!(writer, "- {language}: {count}")?;
    }

    Ok(())
}

pub(super) fn write_search_output(
    writer: &mut dyn Write,
    response: &SearchRepoResponse,
) -> Result<()> {
    writeln!(writer, "query: {}", response.query)?;
    writeln!(writer, "repo_root: {}", response.repo_root.display())?;
    writeln!(writer, "branch: {}", response.branch)?;
    writeln!(writer, "total_results: {}", response.total_results)?;

    if response.results.is_empty() {
        writeln!(writer, "no_results: true")?;
        return Ok(());
    }

    for result in &response.results {
        let display_path = display_path(&response.repo_root, &result.path);
        writeln!(
            writer,
            "- {} [{}] {}:{} score={:.3} match={}",
            result.name, result.kind, display_path, result.line, result.score, result.match_type
        )?;
        writeln!(writer, "  signature: {}", result.signature.trim())?;

        if let Some(doc) = result.doc.as_deref().filter(|doc| !doc.trim().is_empty()) {
            writeln!(writer, "  doc: {}", single_line(doc))?;
        }

        writeln!(writer, "  snippet: {}", single_line(&result.snippet))?;
    }

    Ok(())
}

pub(super) fn write_repomap_output(writer: &mut dyn Write, response: &RepoMap) -> Result<()> {
    writeln!(writer, "repo_root: {}", response.repo_root.display())?;
    writeln!(writer, "branch: {}", response.branch)?;
    writeln!(writer, "modules: {}", response.modules.len())?;

    for module in &response.modules {
        let module_path = display_path(&response.repo_root, &module.path);
        writeln!(writer, "\n## {}", module.name)?;
        writeln!(writer, "path: {module_path}")?;

        if !module.files.is_empty() {
            writeln!(writer, "files:")?;
            for file in &module.files {
                writeln!(writer, "  - {file}")?;
            }
        }

        if !module.symbols.is_empty() {
            writeln!(writer, "symbols:")?;
            for sym in &module.symbols {
                writeln!(
                    writer,
                    "  - {} [{}] {}:{} {}",
                    sym.name,
                    sym.kind,
                    sym.file,
                    sym.line,
                    sym.signature.trim()
                )?;
            }
        }

        if !module.depends_on.is_empty() {
            writeln!(writer, "depends_on: {}", module.depends_on.join(", "))?;
        }
    }

    Ok(())
}

fn display_path(repo_root: &Path, path: &Path) -> String {
    if path.is_absolute() {
        path.strip_prefix(repo_root)
            .map(|relative| relative.display().to_string())
            .unwrap_or_else(|_| path.display().to_string())
    } else {
        path.display().to_string()
    }
}

fn single_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}
