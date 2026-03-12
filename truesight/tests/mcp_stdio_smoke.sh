#!/usr/bin/env bash
set -euo pipefail

stderr_file=$(mktemp)
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -f "$stderr_file"
}
trap cleanup EXIT

coproc SERVER { nix develop --command cargo run --quiet -p truesight -- mcp 2>"$stderr_file"; }

send_message() {
  local body="$1"
  printf '%s\n' "$body" >&"${SERVER[1]}"
}

read_message() {
  local body
  if ! IFS= read -r body <&"${SERVER[0]}"; then
    echo "missing JSON-RPC response" >&2
    cat "$stderr_file" >&2 || true
    return 1
  fi
  printf '%s' "$body"
}

assert_contains() {
  local haystack="$1"
  local needle="$2"
  if [[ "$haystack" != *"$needle"* ]]; then
    echo "expected response to contain: $needle" >&2
    printf '%s\n' "$haystack" >&2
    cat "$stderr_file" >&2 || true
    exit 1
  fi
}

initialize='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"truesight-smoke","version":"0.1.0"}}}'
initialized='{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}'
list_tools='{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

send_message "$initialize"
initialize_response=$(read_message)

send_message "$initialized"
send_message "$list_tools"
tools_response=$(read_message)

assert_contains "$initialize_response" '"capabilities":{"tools":{}'
assert_contains "$initialize_response" 'Truesight indexes local repositories'
assert_contains "$tools_response" '"name":"index_repo"'
assert_contains "$tools_response" '"name":"repo_map"'
assert_contains "$tools_response" '"name":"search_repo"'

printf 'initialize ok\n'
printf 'tools/list ok\n'
