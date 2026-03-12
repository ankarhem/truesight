use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};
use truesight_core::{Result, TruesightError};

pub const DEFAULT_BRANCH: &str = "default";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoContext {
    pub repo_root: PathBuf,
    pub repo_id: String,
    pub branch: String,
    pub last_commit_sha: Option<String>,
}

pub fn detect_repo_context(root: &Path) -> Result<RepoContext> {
    let repo_root = root.canonicalize()?;
    let last_commit_sha = git_output(&repo_root, &["rev-parse", "HEAD"]).ok();

    Ok(RepoContext {
        repo_id: generate_repo_id(&repo_root)?,
        branch: detect_branch_from_root(&repo_root, last_commit_sha.as_deref())?,
        repo_root,
        last_commit_sha,
    })
}

pub fn detect_branch(root: &Path) -> Result<String> {
    let repo_root = root.canonicalize()?;
    let last_commit_sha = git_output(&repo_root, &["rev-parse", "HEAD"]).ok();
    detect_branch_from_root(&repo_root, last_commit_sha.as_deref())
}

pub fn generate_repo_id(repo_root: &Path) -> Result<String> {
    let repo_root = repo_root.canonicalize()?;
    let mut hasher = Sha256::new();
    hasher.update(repo_root.to_string_lossy().as_bytes());
    let digest = hasher.finalize();

    let mut hex = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        let _ = write!(&mut hex, "{byte:02x}");
    }

    Ok(hex)
}

fn detect_branch_from_root(root: &Path, last_commit_sha: Option<&str>) -> Result<String> {
    match git_output(root, &["rev-parse", "--abbrev-ref", "HEAD"]) {
        Ok(value) if value == "HEAD" => Ok(detached_branch_name(last_commit_sha)),
        Ok(value) if !value.is_empty() => Ok(value.to_string()),
        Ok(_) => Ok(DEFAULT_BRANCH.to_string()),
        Err(TruesightError::Git(_)) => Ok(DEFAULT_BRANCH.to_string()),
        Err(error) => Err(error),
    }
}

fn detached_branch_name(last_commit_sha: Option<&str>) -> String {
    last_commit_sha
        .map(str::trim)
        .filter(|sha| !sha.is_empty())
        .map(|sha| format!("detached-{sha}"))
        .unwrap_or_else(|| DEFAULT_BRANCH.to_string())
}

fn git_output(root: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(["-C"])
        .arg(root)
        .args(args)
        .output()?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(TruesightError::Git(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::process::Command;

    use tempfile::TempDir;

    use super::{DEFAULT_BRANCH, detect_branch, detect_repo_context, generate_repo_id};

    #[test]
    fn detect_branch_returns_named_branch_for_git_repo() {
        let temp = git_repo();

        let branch = detect_branch(temp.path()).expect("branch detection should succeed");

        assert_eq!(branch, "main");
    }

    #[test]
    fn detect_branch_returns_detached_commit_partition() {
        let temp = git_repo();
        let commit_sha = git_stdout(temp.path(), &["rev-parse", "HEAD"]);
        run_git(temp.path(), &["checkout", "--detach", "HEAD"]);

        let branch = detect_branch(temp.path()).expect("detached branch detection should succeed");

        assert_eq!(branch, format!("detached-{commit_sha}"));
    }

    #[test]
    fn detect_branch_returns_default_for_non_git_directory() {
        let temp = TempDir::new().expect("temp dir should exist");
        fs::write(temp.path().join("lib.rs"), "pub fn helper() {}")
            .expect("file write should succeed");

        let branch = detect_branch(temp.path()).expect("non-git branch detection should succeed");

        assert_eq!(branch, DEFAULT_BRANCH);
    }

    #[test]
    fn repo_context_generates_stable_repo_id_hash() {
        let temp = git_repo();

        let first = detect_repo_context(temp.path()).expect("context should resolve");
        let second = detect_repo_context(temp.path()).expect("context should resolve twice");
        let other = TempDir::new().expect("other temp dir should exist");
        fs::create_dir_all(other.path().join("nested")).expect("other dir should exist");
        let other_repo_id = generate_repo_id(&other.path().join("nested"))
            .expect("repo id should hash absolute path");

        assert_eq!(first.repo_id, second.repo_id);
        assert_eq!(first.repo_id.len(), 16);
        assert_ne!(first.repo_id, other_repo_id);
        assert_eq!(
            first.repo_root,
            temp.path()
                .canonicalize()
                .expect("canonical path should exist")
        );
    }

    fn git_repo() -> TempDir {
        let temp = TempDir::new().expect("temp dir should exist");
        run_git(temp.path(), &["init", "-b", "main"]);
        fs::create_dir_all(temp.path().join("src")).expect("src dir should exist");
        fs::write(
            temp.path().join("src/lib.rs"),
            "pub fn alpha() -> bool { true }\n",
        )
        .expect("source write should succeed");
        git_commit_all(temp.path(), "initial");
        temp
    }

    fn git_commit_all(root: &Path, message: &str) {
        run_git(root, &["add", "."]);
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
            "git commit failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    fn git_stdout(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(["-C"])
            .arg(root)
            .args(args)
            .output()
            .expect("git command should run");
        assert!(
            output.status.success(),
            "git command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8_lossy(&output.stdout).trim().to_string()
    }

    fn run_git(root: &Path, args: &[&str]) {
        let output = Command::new("git")
            .args(["-C"])
            .arg(root)
            .args(args)
            .output()
            .expect("git command should run");
        assert!(
            output.status.success(),
            "git command failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}
