use ignore::WalkBuilder;
use std::cmp::Ordering;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};
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
        let discovered = Arc::new(Mutex::new(Vec::new()));
        let errors = Arc::new(Mutex::new(Vec::new()));
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
            let discovered = Arc::clone(&discovered);
            let errors = Arc::clone(&errors);
            Box::new(move |entry| {
                match entry {
                    Ok(entry) => {
                        if !entry
                            .file_type()
                            .is_some_and(|file_type| file_type.is_file())
                        {
                            return ignore::WalkState::Continue;
                        }

                        match classify_entry(entry.path(), max_file_size) {
                            Ok(Some(file)) => {
                                discovered
                                    .lock()
                                    .expect("walker result lock poisoned")
                                    .push(file);
                            }
                            Ok(None) => {}
                            Err(error) => {
                                errors
                                    .lock()
                                    .expect("walker error lock poisoned")
                                    .push(error);
                            }
                        }
                    }
                    Err(error) => errors
                        .lock()
                        .expect("walker error lock poisoned")
                        .push(TruesightError::Index(error.to_string())),
                }

                ignore::WalkState::Continue
            })
        });

        if let Some(error) = errors.lock().expect("walker error lock poisoned").pop() {
            return Err(error);
        }

        let mut files = Arc::try_unwrap(discovered)
            .expect("walker result still shared")
            .into_inner()
            .expect("walker result lock poisoned");
        files.sort_by(compare_discovered_files);
        Ok(files)
    }
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
mod tests {
    use super::*;
    use std::collections::BTreeSet;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn walk_rust_fixture_returns_source_files_and_skips_target() {
        let fixture_root = fixture_path("rust-fixture");

        let files = FileWalker::new()
            .walk(&fixture_root)
            .expect("walk should succeed");

        let discovered = relative_paths(&fixture_root, &files);
        assert!(discovered.contains("src/lib.rs"));
        assert!(discovered.contains("src/utils.rs"));
        assert!(!discovered.iter().any(|path| path.starts_with("target/")));
        assert!(!discovered.iter().any(|path| path.starts_with(".git/")));
        assert_eq!(
            files
                .iter()
                .filter(|file| file.language == Language::Rust)
                .count(),
            2
        );
        assert!(files.iter().all(|file| file.size > 0));
    }

    #[test]
    fn walk_ts_fixture_returns_typescript_source_files() {
        let fixture_root = fixture_path("ts-fixture");

        let files = FileWalker::new()
            .walk(&fixture_root)
            .expect("walk should succeed");

        let discovered = relative_paths(&fixture_root, &files);
        assert_eq!(
            discovered,
            BTreeSet::from(["src/index.ts".to_string(), "src/utils.ts".to_string()])
        );
        assert!(
            files
                .iter()
                .all(|file| file.language == Language::TypeScript)
        );
    }

    #[test]
    fn walk_csharp_fixture_returns_csharp_source_files() {
        let fixture_root = fixture_path("csharp-fixture");

        let files = FileWalker::new()
            .walk(&fixture_root)
            .expect("walk should succeed");

        let discovered = relative_paths(&fixture_root, &files);
        assert_eq!(
            discovered,
            BTreeSet::from([
                "Models/User.cs".to_string(),
                "Program.cs".to_string(),
                "Services/AuthService.cs".to_string(),
            ])
        );
        assert!(files.iter().all(|file| file.language == Language::CSharp));
    }

    #[test]
    fn walk_skips_hardcoded_node_modules_directory() {
        let temp_dir = temp_repo();
        fs::write(temp_dir.path().join("package.json"), "{}\n")
            .expect("package.json write should succeed");
        fs::create_dir_all(temp_dir.path().join("src")).expect("src dir should exist");
        fs::create_dir_all(temp_dir.path().join("node_modules/left-pad"))
            .expect("node_modules dir should exist");
        fs::write(
            temp_dir.path().join("src/index.ts"),
            "export const answer = 42;\n",
        )
        .expect("src file write should succeed");
        fs::write(
            temp_dir.path().join("src/view.tsx"),
            "export const View = () => <div />;\n",
        )
        .expect("tsx file write should succeed");
        fs::write(
            temp_dir.path().join("node_modules/left-pad/index.ts"),
            "export const hidden = true;\n",
        )
        .expect("node_modules file write should succeed");

        let files = FileWalker::new()
            .walk(temp_dir.path())
            .expect("walk should succeed");

        let discovered = relative_paths(temp_dir.path(), &files);
        assert_eq!(
            discovered,
            BTreeSet::from(["src/index.ts".to_string(), "src/view.tsx".to_string()])
        );
        assert!(
            files
                .iter()
                .all(|file| file.language == Language::TypeScript)
        );
    }

    #[test]
    fn walk_respects_gitignore_rules() {
        let temp_dir = temp_repo();
        fs::create_dir_all(temp_dir.path().join("src")).expect("src dir should exist");
        fs::write(temp_dir.path().join(".gitignore"), "ignored.ts\nsubdir/\n")
            .expect("gitignore write should succeed");
        fs::write(
            temp_dir.path().join("src/kept.ts"),
            "export const kept = true;\n",
        )
        .expect("kept file write should succeed");
        fs::write(
            temp_dir.path().join("ignored.ts"),
            "export const ignored = true;\n",
        )
        .expect("ignored file write should succeed");
        fs::create_dir_all(temp_dir.path().join("subdir")).expect("subdir should exist");
        fs::write(
            temp_dir.path().join("subdir/skipped.ts"),
            "export const skipped = true;\n",
        )
        .expect("skipped file write should succeed");

        let files = FileWalker::new()
            .walk(temp_dir.path())
            .expect("walk should succeed");

        let discovered = relative_paths(temp_dir.path(), &files);
        assert_eq!(discovered, BTreeSet::from(["src/kept.ts".to_string()]));
    }

    #[test]
    fn walk_skips_binary_files_even_when_extension_matches() {
        let temp_dir = temp_repo();
        fs::write(temp_dir.path().join("text.rs"), "pub fn text() {}\n")
            .expect("text file write should succeed");
        fs::write(
            temp_dir.path().join("binary.ts"),
            b"export const nope = true;\0\x01",
        )
        .expect("binary file write should succeed");

        let files = FileWalker::new()
            .walk(temp_dir.path())
            .expect("walk should succeed");

        let discovered = relative_paths(temp_dir.path(), &files);
        assert_eq!(discovered, BTreeSet::from(["text.rs".to_string()]));
    }

    #[test]
    fn walk_skips_files_over_max_size_threshold() {
        let temp_dir = temp_repo();
        fs::write(temp_dir.path().join("small.rs"), "pub fn small() {}\n")
            .expect("small file write should succeed");
        fs::write(temp_dir.path().join("large.rs"), "a".repeat(128))
            .expect("large file write should succeed");

        let files = FileWalker::with_max_file_size(64)
            .walk(temp_dir.path())
            .expect("walk should succeed");

        let discovered = relative_paths(temp_dir.path(), &files);
        assert_eq!(discovered, BTreeSet::from(["small.rs".to_string()]));
    }

    #[test]
    fn language_from_path_maps_supported_extensions() {
        assert_eq!(
            Language::from_path(Path::new("src/lib.rs")),
            Some(Language::Rust)
        );
        assert_eq!(
            Language::from_path(Path::new("src/index.ts")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            Language::from_path(Path::new("src/view.tsx")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            Language::from_path(Path::new("src/Program.cs")),
            Some(Language::CSharp)
        );
        assert_eq!(Language::from_path(Path::new("README.md")), None);
    }

    fn fixture_path(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("tests")
            .join("fixtures")
            .join(name)
    }

    fn relative_paths(root: &Path, files: &[DiscoveredFile]) -> BTreeSet<String> {
        files
            .iter()
            .map(|file| {
                file.path
                    .strip_prefix(root)
                    .expect("walk result should remain under root")
                    .to_string_lossy()
                    .replace('\\', "/")
            })
            .collect()
    }

    fn temp_repo() -> TempDir {
        let temp_dir = TempDir::new().expect("temp dir should be created");
        fs::create_dir_all(temp_dir.path().join(".git")).expect("git dir should exist");
        temp_dir
    }
}
