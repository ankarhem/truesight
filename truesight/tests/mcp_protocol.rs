use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use serde_json::{Value, json};
use tempfile::NamedTempFile;
use truesight_db::Database;

#[test]
fn mcp_initialize_and_list_tools_reports_exact_contract() {
    let mut server = McpServer::spawn();

    let initialize = server.initialize();
    let instructions = initialize["result"]["instructions"]
        .as_str()
        .expect("instructions should be a string");
    assert_eq!(initialize["jsonrpc"], "2.0");
    assert_eq!(initialize["id"], 1);
    assert_eq!(initialize["result"]["protocolVersion"], "2025-03-26");
    assert_eq!(initialize["result"]["capabilities"]["tools"], json!({}));
    assert_instruction_contract(instructions);

    server.notify_initialized();

    let tools = server.request(2, "tools/list", json!({}));
    assert_eq!(tools["jsonrpc"], "2.0");
    assert_eq!(tools["id"], 2);

    let listed_tools = tools["result"]["tools"]
        .as_array()
        .expect("tools/list should return an array");
    let mut names = listed_tools
        .iter()
        .map(|tool| {
            tool["name"]
                .as_str()
                .expect("tool name should be a string")
                .to_string()
        })
        .collect::<Vec<_>>();
    names.sort();

    assert_eq!(names, vec!["index_repo", "repo_map", "search_repo"]);
    assert_eq!(listed_tools.len(), 3);
    for tool in listed_tools {
        assert!(
            tool.get("inputSchema").is_some(),
            "tool should publish inputSchema"
        );
        assert!(
            tool.get("outputSchema").is_some(),
            "tool should publish outputSchema"
        );
        assert_no_unsupported_integer_formats(&tool["inputSchema"]);
        assert_no_unsupported_integer_formats(&tool["outputSchema"]);
    }
}

fn assert_instruction_contract(instructions: &str) {
    assert!(
        instructions.contains("Recommended agent workflow:"),
        "instructions should include workflow guidance: {instructions}"
    );

    for required_phrase in ["`search_repo`", "`repo_map`", "`index_repo`", "`grep`"] {
        assert!(
            instructions.contains(required_phrase),
            "instructions should mention {required_phrase}: {instructions}"
        );
    }
}

#[test]
fn mcp_index_and_repo_map_tool_calls_work_over_real_stdio_transport() {
    let fixture = rust_fixture_repo();
    cleanup_repo_database(&fixture);

    let mut server = McpServer::spawn();
    server.initialize();
    server.notify_initialized();

    let index_response = server.request(
        2,
        "tools/call",
        json!({
            "name": "index_repo",
            "arguments": {
                "path": fixture.display().to_string(),
                "full": true,
            }
        }),
    );
    assert_eq!(index_response["jsonrpc"], "2.0");
    assert_eq!(index_response["id"], 2);
    assert_eq!(index_response["result"]["isError"], false);

    let index_structured = index_response["result"]["structuredContent"].clone();
    let index_text = index_response["result"]["content"][0]["text"]
        .as_str()
        .expect("tool text content should be present");
    let parsed_index_text: Value =
        serde_json::from_str(index_text).expect("tool text should be valid JSON");
    assert_eq!(parsed_index_text, index_structured);
    assert_eq!(index_structured["status"], "completed");
    assert_eq!(index_structured["repo_root"], fixture.display().to_string());
    assert_eq!(index_structured["branch"], "main");
    assert_eq!(index_structured["stats"]["files_scanned"], 2);
    assert_eq!(index_structured["stats"]["files_indexed"], 2);
    assert_eq!(index_structured["stats"]["files_skipped"], 0);
    assert_eq!(index_structured["stats"]["symbols_extracted"], 12);
    assert_eq!(index_structured["stats"]["chunks_embedded"], 12);
    assert_eq!(index_structured["languages"]["rust"], 2);

    let repo_map_response = server.request(
        3,
        "tools/call",
        json!({
            "name": "repo_map",
            "arguments": {
                "path": fixture.display().to_string(),
                "filter": "src/lib.rs",
            }
        }),
    );
    assert_eq!(repo_map_response["result"]["isError"], false);

    let repo_map_structured = repo_map_response["result"]["structuredContent"].clone();
    let repo_map_text = repo_map_response["result"]["content"][0]["text"]
        .as_str()
        .expect("repo_map text content should be present");
    let parsed_repo_map_text: Value =
        serde_json::from_str(repo_map_text).expect("repo_map text should be valid JSON");
    assert_eq!(parsed_repo_map_text, repo_map_structured);
    assert_eq!(
        repo_map_structured["repo_root"],
        fixture.display().to_string()
    );
    assert_eq!(repo_map_structured["branch"], "main");

    let modules = repo_map_structured["modules"]
        .as_array()
        .expect("repo_map should return modules");
    assert_eq!(modules.len(), 1);
    assert_eq!(modules[0]["name"], "src");
    assert_eq!(modules[0]["path"], "src");
    assert_eq!(modules[0]["files"], json!(["lib.rs"]));

    let symbol_names = modules[0]["symbols"]
        .as_array()
        .expect("module symbols should be an array")
        .iter()
        .map(|symbol| {
            symbol["name"]
                .as_str()
                .expect("symbol name should be a string")
                .to_string()
        })
        .collect::<Vec<_>>();
    assert_eq!(symbol_names.len(), 9);
    assert!(symbol_names.contains(&String::from("User")));
    assert!(symbol_names.contains(&String::from("ValidationError")));
    assert!(symbol_names.contains(&String::from("Validatable")));
    assert!(symbol_names.contains(&String::from("calculate_checksum")));
    assert!(!symbol_names.contains(&String::from("format_name")));

    let user_symbol = modules[0]["symbols"]
        .as_array()
        .expect("module symbols should be an array")
        .iter()
        .find(|symbol| symbol["name"] == "User")
        .expect("repo_map should include User symbol");
    assert_eq!(user_symbol["kind"], "struct");
    assert_eq!(user_symbol["line"], 8);
    assert_eq!(user_symbol["file"], "lib.rs");
    assert_eq!(user_symbol["signature"], "pub struct User");

    cleanup_repo_database(&fixture);
}

#[test]
fn mcp_search_stays_consistent_during_full_reindex() {
    let repo = workspace_repo();
    cleanup_repo_database(&repo);

    let mut server = McpServer::spawn();
    server.initialize();
    server.notify_initialized();

    server.send_request(
        2,
        "tools/call",
        json!({
            "name": "index_repo",
            "arguments": {
                "path": repo.display().to_string(),
                "full": true,
            }
        }),
    );

    for request_id in 3..13 {
        server.send_request(
            request_id,
            "tools/call",
            json!({
                "name": "search_repo",
                "arguments": {
                    "path": repo.display().to_string(),
                    "query": "mcp schema format uint32",
                    "limit": 5,
                }
            }),
        );
    }

    let mut search_totals = Vec::new();
    for _ in 0..11 {
        let response = server.read_response();
        assert_eq!(response["jsonrpc"], "2.0");

        if response["id"] == 2 {
            assert_eq!(response["result"]["isError"], false);
            continue;
        }

        assert_eq!(
            response["result"]["isError"], false,
            "unexpected overlapping search error response: {response}"
        );
        search_totals.push(
            response["result"]["structuredContent"]["total_results"]
                .as_u64()
                .expect("search result count should be present"),
        );
    }

    assert_eq!(search_totals.len(), 10);
    assert!(
        search_totals.iter().all(|count| *count > 0),
        "overlapping search should not observe an empty index: {search_totals:?}"
    );

    cleanup_repo_database(&repo);
}

struct McpServer {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_file: NamedTempFile,
}

impl McpServer {
    fn spawn() -> Self {
        let stderr_file = NamedTempFile::new().expect("stderr temp file should be created");
        let stderr = stderr_file
            .reopen()
            .expect("stderr temp file should reopen");

        let mut child = Command::new(env!("CARGO_BIN_EXE_truesight"))
            .arg("mcp")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(stderr))
            .spawn()
            .expect("truesight mcp should spawn");

        let stdin = child.stdin.take().expect("child stdin should be piped");
        let stdout = child.stdout.take().expect("child stdout should be piped");

        Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            stderr_file,
        }
    }

    fn initialize(&mut self) -> Value {
        self.request(
            1,
            "initialize",
            json!({
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {
                    "name": "truesight-mcp-test",
                    "version": "0.1.0",
                }
            }),
        )
    }

    fn notify_initialized(&mut self) {
        self.send(&json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }));
    }

    fn request(&mut self, id: u64, method: &str, params: Value) -> Value {
        self.send_request(id, method, params);
        self.read_response()
    }

    fn send_request(&mut self, id: u64, method: &str, params: Value) {
        self.send(&json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }));
    }

    fn read_response(&mut self) -> Value {
        let mut line = String::new();
        let bytes = self
            .stdout
            .read_line(&mut line)
            .expect("MCP server should write a response line");
        assert!(
            bytes > 0,
            "server exited before responding: {}",
            self.stderr_contents()
        );

        serde_json::from_str(line.trim_end()).unwrap_or_else(|error| {
            panic!(
                "server returned invalid JSON: {error}; stderr: {}",
                self.stderr_contents()
            )
        })
    }

    fn send(&mut self, message: &Value) {
        let payload = serde_json::to_string(message).expect("JSON-RPC message should serialize");
        writeln!(self.stdin, "{payload}").expect("MCP server stdin should accept request");
        self.stdin.flush().expect("MCP server stdin should flush");
    }

    fn stderr_contents(&self) -> String {
        fs::read_to_string(self.stderr_file.path()).unwrap_or_default()
    }
}

impl Drop for McpServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn rust_fixture_repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("rust-fixture")
        .canonicalize()
        .expect("fixture path should resolve")
}

fn workspace_repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .expect("workspace path should resolve")
}

fn cleanup_repo_database(repo_root: &Path) {
    if let Ok(db_path) = Database::db_path_for_repo(repo_root) {
        let _ = fs::remove_file(&db_path);
        let _ = fs::remove_file(format!("{}-wal", db_path.display()));
        let _ = fs::remove_file(format!("{}-shm", db_path.display()));
    }
}

fn assert_no_unsupported_integer_formats(value: &Value) {
    match value {
        Value::Object(map) => {
            if let Some(format) = map.get("format").and_then(Value::as_str) {
                assert!(
                    format != "uint32" && format != "uint",
                    "schema should not contain unsupported integer format `{format}`: {value}"
                );
            }

            for nested in map.values() {
                assert_no_unsupported_integer_formats(nested);
            }
        }
        Value::Array(items) => {
            for item in items {
                assert_no_unsupported_integer_formats(item);
            }
        }
        _ => {}
    }
}
