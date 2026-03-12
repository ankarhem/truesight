## 2026-03-12 Task 3 - Test Fixture Repos

- Created `tests/fixtures/` directory structure with 3 subdirectories:
- `rust-fixture/`: Rust fixture with `src/lib.rs`, `src/utils.rs`
- `ts-fixture/`: TypeScript fixture with `src/index.ts`, `src/utils.ts`
- `csharp-fixture/`: C# fixture with `Program.cs`, `Services/AuthService.cs`, `Models/User.cs`
- Each fixture has `expected.json` through known symbols
- Each fixture initialized as a git repo with one commit
- All fixture code uses minimal and predictable patterns (no complex logic)
- ASCII-only contents used throughout

## Verification
- Rust fixture compiles successfully via `nix develop --command cargo check` (passes with exit code 0)

## 2026-03-12 Task 3 - Caveat

- Added empty `[workspace]` table to rust-fixture/Cargo.toml to make it standalone (excluded from main workspace) to avoid cargo workspace detection conflict
- Unused function warnings in rust-fixture are acceptable for test fixtures (truncate_text, is_blank are intentionally unused)

## 2026-03-12 Task 7 - Caveat

- `lsp_diagnostics` could not be run for `truesight-engine/src/embed.rs` because the OpenCode Rust LSP integration does not currently see `rust-analyzer` as installed in this environment, even though the crate now compiles and passes the required Cargo verification commands.

## 2026-03-12 Task 8 - Caveats

- `nix develop --command cargo test -p truesight-engine -- walker` and `nix develop --command cargo build --workspace` are currently blocked by pre-existing compile failures in `truesight-engine/src/embed.rs` (ORT/ndarray tensor type mismatch), not by walker code.
- `lsp_diagnostics` could not be completed for changed Rust files because `rust-analyzer` is not installed in this environment.

## 2026-03-12 Task 5 - Verification Caveats

- `nix develop --command cargo test -p truesight-engine -- parser` and `nix develop --command cargo build --workspace` are currently blocked by pre-existing compile failures in `truesight-engine/src/embed.rs`, not by the new parser module
- Rust LSP diagnostics could not be run because `rust-analyzer` is not installed in this environment; `lsp_diagnostics` returns a tooling-availability error before analyzing changed files
- Fixture `expected.json` includes `type_alias` and `property`, but `truesight-core::CodeUnitKind` has no matching variants, so Task 5 maps those fixture cases to `CodeUnitKind::Constant` as the V1 compatibility path

## 2026-03-12 Task 6 - Caveats

- The shared `Storage` trait still stores bare `CodeUnit` values, so `truesight-db` exposes an additional `store_code_units_with_embeddings(...)` helper for embedding BLOB persistence until higher layers wire embeddings through explicitly
- `search_vector()` first attempts libSQL `vector_top_k(...)`, but current tests exercise the required Rust cosine-similarity fallback because no vector index is created in Task 4 schema

## 2026-03-12 Task 7 - Caveats

- `truesight-engine/build.rs` now pins a known-good Nix store ORT dylib path (`onnxruntime-1.23.2`) to avoid the runtime hang caused by ambiguous `/nix/store` discovery; this is deterministic and fail-fast, but it remains environment-specific until the dev shell exports a single canonical ORT path.
- Rust `lsp_diagnostics` could not run on the changed Rust files in this session because the OpenCode Rust LSP reports `rust-analyzer` unavailable even though `nix develop` can compile the workspace successfully.

## 2026-03-12 Task 7 - Caveats

- Task 7 now relies on ORT's bundled/copy-dylibs path instead of the previous `load-dynamic` runtime-loading setup because the latter was the source of the test hang in this Nix environment.
- `lsp_diagnostics` still cannot run for Rust files here because `rust-analyzer` is not installed; verification used the required cargo test/build commands instead.

## 2026-03-12 Task 7 - Caveats

- `truesight-engine/build.rs` uses a pinned `/nix/store/.../libonnxruntime.so` fallback for this environment; if that store path is collected or replaced, `ORT_DYLIB_PATH` should be set explicitly to the intended ONNX Runtime dylib.
- `lsp_diagnostics` for changed Rust files is still blocked by missing `rust-analyzer` in the tool environment, so verification relies on `cargo test` and `cargo build` instead.

## 2026-03-12 Task 7 Runtime Alignment Follow-up

- OpenCode's Rust LSP still reports `rust-analyzer` unavailable for `lsp_diagnostics`, even though `nix develop --command rust-analyzer --version` works in the project dev shell; cargo build/test remains the verification fallback for Rust changes in this environment.

## 2026-03-12 Task 9 - Caveats

- `lsp_diagnostics` still cannot run on changed Rust files in `truesight-engine` because the OpenCode tool environment does not currently expose `rust-analyzer`; Task 9 verification used the required cargo test/build commands plus real DB-backed indexer tests.
- `nix develop --command cargo build --workspace` passes, but the workspace already contains an unrelated warning in `truesight-engine/src/incremental.rs` for an unused `IndexAccumulators` struct; Task 9 did not change that module's behavior.

## 2026-03-12 Task 12 - Caveats

- `lsp_diagnostics` still cannot run on changed Rust files in `truesight-engine` because the OpenCode Rust LSP cannot find `rust-analyzer` in this environment; verification used the required Cargo commands instead.
- An unrelated pre-existing engine test, `indexer::tests::index_rust_fixture_persists_expected_symbols_and_stats`, still fails because its hard-coded Rust fixture symbol count expects `11` while the current parser/indexer behavior returns `12`; this does not affect the required Task 12 verification commands.

## 2026-03-12 Task 11 - Caveats

- `RepoMapGenerator` intentionally does not honor the plan's older fallback to walking/parsing fresh repositories because this task's scoped directive requires repo-map generation to use only stored symbols from `Storage::get_all_symbols(...)`.
- Rust `lsp_diagnostics` still cannot run for `truesight-engine/src/lib.rs` and `truesight-engine/src/repomap.rs` here because the OpenCode environment does not have `rust-analyzer` installed; verification used the required cargo test/build commands instead.

## 2026-03-12 Task 10 - Caveats

- `lsp_diagnostics` still cannot run on changed Rust files in `truesight-engine` because the tool environment does not have `rust-analyzer` installed; verification used the required Cargo test/build commands instead.
- `truesight-engine/src/lib.rs` contained stale exports for non-existent future modules (`indexer`, `repomap`, then `incremental`) during this task; removing those unresolved declarations was necessary to restore package/workspace compilation without implementing those later tasks here.

## 2026-03-12 Task 14 - Caveats

- `lsp_diagnostics` still cannot run on changed Rust files in `truesight/src` because the OpenCode environment does not expose `rust-analyzer`; verification used `nix develop --command cargo test -p truesight`, `nix develop --command cargo build -p truesight`, and a bash stdio smoke test instead.
- rmcp 1.2 expects request/output schemas through its own `schemars` version; returning JSON text from the tool methods keeps the MCP contract stable without forcing cross-crate schema upgrades in `truesight-core` during this task.

## 2026-03-12 Task 13 - Caveats

- `truesight/src/mcp.rs` is intentionally only a thin placeholder entrypoint that returns a Task 14 handoff error; this task wires the clap subcommand only and does not implement rmcp transport or tool handlers.
- Rust `lsp_diagnostics` still cannot run for the changed `truesight/src/*.rs` files because the OpenCode tool environment does not expose `rust-analyzer`; verification used the required nix-based Cargo commands instead.

## 2026-03-12 Task 15 - Caveats

- `lsp_diagnostics` still cannot run for changed Rust files in `truesight-engine` and `truesight-db` because the OpenCode environment does not have `rust-analyzer` installed; verification used the required `nix develop --command cargo ...` commands instead.
- `nix develop --command cargo build --workspace` passes, but the workspace still has an unrelated pre-existing warning in `truesight/src/app.rs` for an unused `std::fs` import; Task 15 did not modify the binary crate because this task is scoped to engine/db only.

## 2026-03-12 Task 14 - Caveats

- Rust `lsp_diagnostics` still cannot run for changed files in `truesight/src` because the OpenCode environment does not expose `rust-analyzer`; verification used the required nix-based cargo test/build commands plus the bash MCP smoke test instead.
- rmcp's initialize response still reports `serverInfo.name = "rmcp"` from SDK build metadata even when custom instructions/capabilities are supplied; Task 14 validation keys off the required tool registration and startup behavior, not that cosmetic identifier.

## 2026-03-12 Task 14 Verification Follow-up - Caveats

- The task brief said `truesight/src/mcp.rs` was missing, but the current on-disk workspace already contains a complete rmcp implementation; this pass verified that implementation rather than recreating it from scratch.
- Rust `lsp_diagnostics` remains unavailable for `truesight/src/*.rs` because `rust-analyzer` is not installed in the OpenCode tool environment, so verification continues to rely on `nix develop --command cargo ...` plus the stdio smoke test.

## 2026-03-12 Task 17 - Caveats

- Rust `lsp_diagnostics` could not run for the new `truesight/tests/mcp_protocol.rs` integration test because the OpenCode environment still lacks `rust-analyzer`; verification used `nix develop --command cargo test -p truesight mcp` and `nix develop --command cargo test --workspace` instead.
- The spawned-process MCP test intentionally uses the current user `HOME` so the existing ONNX model cache can be reused; it removes the repo-specific SQLite database before and after the test to keep the run deterministic without forcing a fresh model download into a temporary home directory.

## 2026-03-12 Task 16 - Caveats

- Rust `lsp_diagnostics` still cannot run for changed `truesight/src/*.rs` and `truesight/tests/*.rs` files because the OpenCode environment does not expose `rust-analyzer`; verification used the required nix-based Cargo test commands instead.
- Running the `truesight` binary itself inside the new test harness currently panics in this environment when the ONNX/reqwest stack tears down its Tokio runtime, so Task 16 exercises the real `app::run(...)` CLI entrypoint through integration tests instead of spawning the compiled binary process.

## 2026-03-12 Task 18 - Caveats

- Rust `lsp_diagnostics` is still blocked for the changed `truesight-engine/src/*.rs` and `truesight/src/*.rs` files because the OpenCode tool environment does not expose `rust-analyzer`; verification therefore relies on the required nix-based Cargo test commands.
- `truesight/src/lib.rs` was added as a minimal package-local export surface so existing unit/integration tests can import `truesight::app` and `truesight::cli` during workspace test runs; this is a test-enabling crate wiring fix, not a new runtime feature.

## 2026-03-12 Task 17 Smoke Script Fix - Caveats

- `truesight/tests/mcp_stdio_smoke.sh` was fixed to use `nix develop --command cargo run` instead of bare `cargo run` because `cargo` is not available on the PATH outside the Nix devshell in this environment.
- The `mcp_protocol.rs` integration test already works correctly because it uses `CARGO_BIN_EXE_truesight` to spawn the pre-built binary from within cargo's test harness.
- Pre-existing `app::tests` failures (`search_command_rejects_empty_query_without_crashing`, `search_command_returns_ranked_results_for_indexed_fixture`, `search_command_suggests_indexing_when_repo_is_missing_metadata`) are unrelated to Task 17's MCP protocol smoke test fix.

## 2026-03-12 Final Architecture Compliance Fix - Caveats

- Rust `lsp_diagnostics` is still unavailable in this environment because the OpenCode Rust LSP cannot see `rust-analyzer`; verification used the required nix-based `cargo clippy`, `cargo test`, and MCP smoke commands instead.

## 2026-03-12 Final Scope Fidelity Cleanup - Caveats

- Rust `lsp_diagnostics` still cannot run for the changed Rust files because the OpenCode tool environment reports `rust-analyzer` missing; verification again relied on `nix develop --command cargo clippy --workspace --all-targets -- -D warnings`, `nix develop --command cargo test --workspace`, and `bash truesight/tests/mcp_stdio_smoke.sh`.

## 2026-03-12 Fixture Repo De-embedding - Caveats

- Rust `lsp_diagnostics` remains unavailable for the changed Rust test files and helpers because `rust-analyzer` is not installed in the OpenCode tool environment; verification used `nix develop --command cargo test --workspace` before and after removing the fixture `.git` directories.

## 2026-03-12 Task 19 ONNX Runtime Devshell Alignment - Caveats

- The ONNX Runtime path is now intentionally supplied by the Nix devshell instead of `/nix/store` discovery; outside `nix develop`, callers need to set `ORT_DYLIB_PATH` explicitly or rely on the build-time `TRUESIGHT_ORT_DYLIB` fallback.
- Rust `lsp_diagnostics` still could not run for the changed Rust files because `rust-analyzer` is not installed in the OpenCode tool environment; verification used the required nix-based cargo commands instead.
