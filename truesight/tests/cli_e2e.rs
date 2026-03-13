use std::fs;
use std::path::Path;
use std::process::Command;

#[path = "test_support/git_fixture.rs"]
mod git_fixture;

use git_fixture::TempGitFixture;

#[test]
fn index_then_search_fixture_via_real_binary() {
    let sandbox = TestSandbox::new("csharp-fixture");

    let index_stdout = sandbox
        .run_cli(["index", sandbox.repo_str(), "--full"])
        .expect("index fixture repository");
    assert!(
        index_stdout.contains("branch: main"),
        "expected main branch in index output: {index_stdout}"
    );
    assert!(
        index_stdout.contains("mode: full"),
        "expected full mode in index output: {index_stdout}"
    );
    assert!(
        index_stdout.contains("files_indexed:"),
        "expected file stats in index output: {index_stdout}"
    );
    assert!(
        index_stdout.contains("symbols_extracted:"),
        "expected symbol stats in index output: {index_stdout}"
    );

    let search_stdout = sandbox
        .run_cli([
            "search",
            "authentication providers",
            "--repo",
            sandbox.repo_str(),
            "--limit",
            "5",
        ])
        .expect("search indexed fixture repository");
    assert!(
        search_stdout.contains("branch: main"),
        "expected main branch in search output: {search_stdout}"
    );
    assert!(
        search_stdout.contains("total_results:"),
        "expected result count in search output: {search_stdout}"
    );
    assert!(
        search_stdout.contains("AuthService"),
        "expected AuthService result in search output: {search_stdout}"
    );
    assert!(
        search_stdout.contains("IAuthProvider"),
        "expected interface result in search output: {search_stdout}"
    );
    assert!(
        search_stdout.contains("match="),
        "expected match metadata in search output: {search_stdout}"
    );
    assert!(
        search_stdout.contains("snippet:"),
        "expected snippet in search output: {search_stdout}"
    );
}

#[test]
fn branch_indexes_remain_isolated_across_checkout_changes() {
    let sandbox = TestSandbox::new("rust-fixture");

    sandbox
        .run_cli(["index", sandbox.repo_str(), "--full"])
        .expect("index main branch");

    let main_stdout = sandbox
        .run_cli([
            "search",
            "formatted display string",
            "--repo",
            sandbox.repo_str(),
            "--limit",
            "5",
        ])
        .expect("search main branch");
    assert!(
        main_stdout.contains("branch: main"),
        "expected main branch search output: {main_stdout}"
    );
    assert!(
        main_stdout.contains("display_name"),
        "expected display_name result before branch change: {main_stdout}"
    );

    sandbox.run_git(["checkout", "-b", "feature/profile-label"]);
    add_branch_only_symbol(sandbox.fixture.path());
    sandbox.run_git(["add", "src/lib.rs"]);
    sandbox.run_git([
        "-c",
        "user.name=Truesight Tests",
        "-c",
        "user.email=tests@example.com",
        "commit",
        "-m",
        "Add feature branch profile label",
    ]);

    sandbox
        .run_cli(["index", sandbox.repo_str(), "--full"])
        .expect("index feature branch");

    let feature_stdout = sandbox
        .run_cli([
            "search",
            "branch_only_label",
            "--repo",
            sandbox.repo_str(),
            "--limit",
            "5",
        ])
        .expect("search feature branch");
    assert!(
        feature_stdout.contains("branch: feature/profile-label"),
        "expected feature branch search output: {feature_stdout}"
    );
    assert!(
        feature_stdout.contains("- branch_only_label ["),
        "expected branch-only symbol on feature branch: {feature_stdout}"
    );

    sandbox.run_git(["checkout", "main"]);

    let isolation_stdout = sandbox
        .run_cli([
            "search",
            "branch_only_label",
            "--repo",
            sandbox.repo_str(),
            "--limit",
            "5",
        ])
        .expect("search main branch after switching back");
    assert!(
        isolation_stdout.contains("branch: main"),
        "expected main branch after checkout: {isolation_stdout}"
    );
    assert!(
        !isolation_stdout.contains("- branch_only_label ["),
        "feature-only symbol leaked into main branch results: {isolation_stdout}"
    );

    let main_again_stdout = sandbox
        .run_cli([
            "search",
            "formatted display string",
            "--repo",
            sandbox.repo_str(),
            "--limit",
            "5",
        ])
        .expect("re-search main branch after feature indexing");
    assert!(
        main_again_stdout.contains("display_name"),
        "expected main branch symbol after returning from feature branch: {main_again_stdout}"
    );
}

struct TestSandbox {
    fixture: TempGitFixture,
}

impl TestSandbox {
    fn new(fixture_name: &str) -> Self {
        Self {
            fixture: TempGitFixture::new(fixture_name),
        }
    }

    fn repo_str(&self) -> &str {
        self.fixture.path_str()
    }

    fn run_cli<I, S>(&self, args: I) -> anyhow::Result<String>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let output = Command::new(env!("CARGO_BIN_EXE_truesight"))
            .args(args.into_iter().map(|arg| arg.as_ref().to_string()))
            .output()
            .expect("truesight command should run");
        assert!(
            output.status.success(),
            "truesight failed with status {:?}\nstdout:\n{}\nstderr:\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(String::from_utf8(output.stdout).expect("cli output should be utf-8"))
    }

    fn run_git<I, S>(&self, args: I)
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        self.fixture.run_git(args);
    }
}

fn add_branch_only_symbol(repo_root: &Path) {
    let lib_rs = repo_root.join("src/lib.rs");
    let mut source = fs::read_to_string(&lib_rs).expect("fixture source should read");
    source.push_str(
        "\n/// Builds a feature branch profile label.\npub fn branch_only_label(name: &str) -> String {\n    format!(\"feature-profile:{}\", name)\n}\n",
    );
    fs::write(&lib_rs, source).expect("fixture source should update");
}
