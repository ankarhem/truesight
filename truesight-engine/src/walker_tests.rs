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
