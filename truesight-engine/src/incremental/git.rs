use std::path::{Path, PathBuf};
use std::process::Command;

use truesight_core::{Language, Result, TruesightError};

use super::ChangeSet;

pub(super) fn is_git_repo(root: &Path) -> bool {
    git_command(root, ["rev-parse", "--git-dir"]).is_ok()
}

pub(super) fn current_commit_sha(root: &Path) -> Result<String> {
    git_command(root, ["rev-parse", "HEAD"])
}

pub(super) fn git_changes(root: &Path, last_sha: &str) -> Result<ChangeSet> {
    let mut changes = ChangeSet {
        added: git_diff_paths(root, last_sha, "A")?,
        modified: git_diff_paths(root, last_sha, "M")?,
        deleted: git_diff_paths(root, last_sha, "D")?,
    };

    changes.normalize();
    Ok(changes)
}

fn git_diff_paths(root: &Path, last_sha: &str, diff_filter: &str) -> Result<Vec<PathBuf>> {
    let diff_filter_arg = format!("--diff-filter={diff_filter}");
    let output = git_command_bytes(
        root,
        &[
            "diff",
            "--name-only",
            "--no-renames",
            diff_filter_arg.as_str(),
            "-z",
            last_sha,
            "HEAD",
        ],
    )?;

    output
        .split(|byte| *byte == 0)
        .filter(|chunk| !chunk.is_empty())
        .filter_map(|chunk| {
            let path = match std::str::from_utf8(chunk) {
                Ok(path) => PathBuf::from(path),
                Err(error) => return Some(Err(TruesightError::Git(error.to_string()))),
            };

            if Language::from_path(&path).is_none() {
                return None;
            }

            Some(Ok(path))
        })
        .collect()
}

fn git_command_bytes(root: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| TruesightError::Git(error.to_string()))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
        return Err(TruesightError::Git(if stderr.is_empty() {
            format!("git command failed: git {}", args.join(" "))
        } else {
            stderr
        }));
    }

    Ok(output.stdout)
}

fn git_command<const N: usize>(root: &Path, args: [&str; N]) -> Result<String> {
    let output = git_command_bytes(root, &args)?;

    String::from_utf8(output)
        .map(|stdout| stdout.trim().to_string())
        .map_err(|error| TruesightError::Git(error.to_string()))
}

#[cfg(test)]
pub(super) fn git_commit_all(root: &Path, message: &str) {
    run_git(root, ["add", "."]);
    let mut command = Command::new("git");
    command
        .args(["commit", "-m", message])
        .current_dir(root)
        .env("GIT_AUTHOR_NAME", "OpenCode")
        .env("GIT_AUTHOR_EMAIL", "opencode@example.com")
        .env("GIT_COMMITTER_NAME", "OpenCode")
        .env("GIT_COMMITTER_EMAIL", "opencode@example.com");
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "git commit failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[cfg(test)]
pub(super) fn run_git<const N: usize>(root: &Path, args: [&str; N]) {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git command failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
