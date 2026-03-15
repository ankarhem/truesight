use crossbeam_channel::{Sender, unbounded};
use ignore::WalkBuilder;
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use truesight_core::{Language, Result, TruesightError};

const BINARY_CHECK_BYTES: usize = 8 * 1024;
const DEFAULT_MAX_FILE_SIZE: u64 = 1024 * 1024;
const EXCLUDED_DIRECTORIES: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    "build",
    "vendor",
    ".direnv",
    "bin",
    "obj",
    "packages",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredFile {
    pub path: PathBuf,
    pub language: Language,
    pub size: u64,
}

#[derive(Debug, Clone)]
pub struct FileWalker {
    max_file_size: u64,
}

impl Default for FileWalker {
    fn default() -> Self {
        Self::new()
    }
}

impl FileWalker {
    pub fn new() -> Self {
        Self {
            max_file_size: DEFAULT_MAX_FILE_SIZE,
        }
    }

    pub fn with_max_file_size(max_file_size: u64) -> Self {
        Self { max_file_size }
    }

    pub fn walk(&self, root: &Path) -> Result<Vec<DiscoveredFile>> {
        let (tx, rx) = unbounded();
        let max_file_size = self.max_file_size;

        let mut builder = WalkBuilder::new(root);
        builder
            .hidden(false)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .parents(true)
            .require_git(false)
            .filter_entry(is_allowed_entry)
            .follow_links(false);

        builder.build_parallel().run(|| {
            let tx = tx.clone();
            Box::new(move |entry| {
                let event = match entry {
                    Ok(entry) => {
                        if !entry
                            .file_type()
                            .is_some_and(|file_type| file_type.is_file())
                        {
                            return ignore::WalkState::Continue;
                        }

                        match classify_entry(entry.path(), max_file_size) {
                            Ok(Some(file)) => Some(WalkEvent::Discovered(file)),
                            Ok(None) => None,
                            Err(error) => Some(WalkEvent::Error(error)),
                        }
                    }
                    Err(error) => Some(WalkEvent::Error(TruesightError::Index(error.to_string()))),
                };

                if let Some(event) = event {
                    send_walk_event(&tx, event);
                }

                ignore::WalkState::Continue
            })
        });
        drop(tx);

        let mut files = Vec::new();
        let mut first_error = None;

        for event in rx {
            match event {
                WalkEvent::Discovered(file) => files.push(file),
                WalkEvent::Error(error) if first_error.is_none() => first_error = Some(error),
                WalkEvent::Error(_) => {}
            }
        }

        if let Some(error) = first_error {
            return Err(error);
        }

        files.sort_by(compare_discovered_files);
        Ok(files)
    }
}

enum WalkEvent {
    Discovered(DiscoveredFile),
    Error(TruesightError),
}

fn send_walk_event(sender: &Sender<WalkEvent>, event: WalkEvent) {
    let _ = sender.send(event);
}

fn compare_discovered_files(left: &DiscoveredFile, right: &DiscoveredFile) -> Ordering {
    left.path
        .cmp(&right.path)
        .then_with(|| left.size.cmp(&right.size))
}

fn is_allowed_entry(entry: &ignore::DirEntry) -> bool {
    if entry.depth() == 0 {
        return true;
    }

    !entry
        .path()
        .components()
        .filter_map(component_name)
        .any(|component| EXCLUDED_DIRECTORIES.contains(&component))
}

fn classify_entry(path: &Path, max_file_size: u64) -> Result<Option<DiscoveredFile>> {
    let Some(language) = Language::from_path(path) else {
        return Ok(None);
    };

    if has_excluded_component(path) {
        return Ok(None);
    }

    let metadata = fs::metadata(path)?;
    let size = metadata.len();
    if size > max_file_size {
        return Ok(None);
    }

    if is_binary_file(path)? {
        return Ok(None);
    }

    Ok(Some(DiscoveredFile {
        path: path.to_path_buf(),
        language,
        size,
    }))
}

fn has_excluded_component(path: &Path) -> bool {
    path.components()
        .filter_map(component_name)
        .any(|component| EXCLUDED_DIRECTORIES.contains(&component))
}

fn component_name(component: Component<'_>) -> Option<&str> {
    match component {
        Component::Normal(name) => name.to_str(),
        _ => None,
    }
}

fn is_binary_file(path: &Path) -> Result<bool> {
    let mut file = File::open(path)?;
    let mut buffer = [0_u8; BINARY_CHECK_BYTES];
    let bytes_read = file.read(&mut buffer)?;
    Ok(buffer[..bytes_read].contains(&0))
}

#[cfg(test)]
#[path = "walker_tests.rs"]
mod tests;
