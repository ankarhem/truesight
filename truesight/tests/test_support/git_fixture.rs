use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

pub struct TempGitFixture {
    _temp_dir: TempDir,
    repo_root: PathBuf,
}

#[allow(dead_code)]
impl TempGitFixture {
    pub fn new(fixture_name: &str) -> Self {
        let temp_dir = tempfile::tempdir().expect("tempdir should exist");
        let repo_root = temp_dir.path().join("repo");

        copy_fixture_dir(&fixture_path(fixture_name), &repo_root);
        init_git_repo(&repo_root, "Fixture snapshot");

        Self {
            _temp_dir: temp_dir,
            repo_root,
        }
    }

    pub fn path(&self) -> &Path {
        &self.repo_root
    }

    pub fn path_buf(&self) -> PathBuf {
        self.repo_root.clone()
    }

    pub fn path_str(&self) -> &str {
        self.repo_root.to_str().expect("repo path should be utf-8")
    }

    pub fn run_git<I, S>(&self, args: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        run_git_in_dir(&self.repo_root, args);
    }
}

pub fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("fixtures")
        .join(name)
}

pub fn init_git_repo(root: &Path, message: &str) {
    init_empty_git_repo(root);
    commit_all(root, message);
}

pub fn init_empty_git_repo(root: &Path) {
    run_git_in_dir(root, ["init", "-b", "main"]);
}

pub fn commit_all(root: &Path, message: &str) {
    run_git_in_dir(root, ["add", "."]);

    let output = Command::new("git")
        .args(["commit", "-m", message])
        .current_dir(root)
        .env("GIT_AUTHOR_NAME", "OpenCode")
        .env("GIT_AUTHOR_EMAIL", "opencode@example.com")
        .env("GIT_COMMITTER_NAME", "OpenCode")
        .env("GIT_COMMITTER_EMAIL", "opencode@example.com")
        .output()
        .expect("git commit should run");
    assert!(
        output.status.success(),
        "git commit failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn copy_fixture_dir(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).expect("fixture destination should exist");

    for entry in fs::read_dir(source).expect("fixture directory should read") {
        let entry = entry.expect("fixture entry should read");
        let file_type = entry.file_type().expect("fixture file type should read");
        let name = entry.file_name();

        if name == ".git" {
            continue;
        }

        let source_path = entry.path();
        let destination_path = destination.join(&name);

        if file_type.is_dir() {
            copy_fixture_dir(&source_path, &destination_path);
        } else if file_type.is_file() {
            fs::copy(&source_path, &destination_path).expect("fixture file should copy");
        }
    }
}

fn run_git_in_dir<I, S>(root: &Path, args: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut command = Command::new("git");
    command.current_dir(root);

    for arg in args {
        command.arg(arg.as_ref());
    }

    let output = command.output().expect("git command should run");
    assert!(
        output.status.success(),
        "git command failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}
