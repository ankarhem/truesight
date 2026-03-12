# Learnings


 <content>
   <content>
3: ## 2026-03-12 Task 1 Fidelity Fixes - Fixed rmcp version from 0.1 to 1.2 (align with plan)
- Fixed tree-sitter version from 0.24 into 0.26 (align with plan's tree-sitter 5.26 generation)
- Updated tree-sitter language crates to 0.24.0, 0.23.2, 0.23.1 (exact patch)
- Updated tokenizers from 0.19 to 0.21.2 (kept ort configuration)
- Re-ran build and test to verify success

   - Removed unnecessary scope creep from .gitignore (Cargo.lock already excluded, pure scope creep)
- Build and tests still pass
</content>

## 2026-03-12 Task 1 Additional Fidelity Fixes

- Removed `Cargo.lock` from .gitignore (not requested by plan, scope creep)
- Removed unused `[workspace.package]` section from root Cargo.toml (crates don't inherit from it, scope creep)
- Verified all dependency versions align with plan:
  - rmcp 1.2 ✓
  - tree-sitter 0.26 ✓
  - tree-sitter-rust 0.24 ✓
  - tree-sitter-typescript 0.23 ✓
  - tree-sitter-c-sharp 0.23 ✓
  - ort 2.0.0-rc.12 with load-dynamic ✓
  - tokenizers 0.19 ✓
- Build and tests pass after all changes

## 2026-03-12 Task 4 Schema + Migrations

- `truesight-db` now tracks the base schema as migration version `1` in `_migrations` instead of only relying on `CREATE IF NOT EXISTS`
- `db_path_for_repo()` is deterministic by hashing the normalized repo root and keeping a sanitized repo-name prefix in the filename
- `libsql::Builder::new_local()` plus a per-connection `PRAGMA busy_timeout = 5000` works for local file-backed schema tests in this workspace

## 2026-03-12 Task 3 Fix

- Fixed invalid Rust syntax in rust-fixture/src/lib.rs: `mod utils::format_name;` was replaced with `mod utils; use crate::utils::format_name;`
- Added empty `[workspace]` table to rust-fixture/Cargo.toml to make it standalone (prevents workspace detection errors)
- Updated expected.json to reflect correct import syntax
- Rust fixture now compiles successfully via `nix develop --command cargo check`
- All three fixture git repos verified with valid commits
- Symbol counts verified: rust=12, ts=14, csharp=18 (all >= 5 minimum)
- Documented symbols verified: all fixtures have >= 2 documented symbols

## 2026-03-12 Task 7 Embedder Fixes

- Re-aligned `truesight-engine` to `ort 2.0.0-rc.12` and `ndarray 0.17.2`, which matches `ort`'s tensor APIs and fixes the earlier ndarray-version conflict.
- `TensorRef::from_array_view(...)` now receives ndarray views from owned arrays, which compiles cleanly with `ort` and preserves the existing mean-pooling plus L2-normalization flow.
- Switched `ort` to use explicit rustls-backed download features instead of the prior dynamic-runtime path workaround so embed tests and workspace builds run cleanly in the Nix devshell.

## 2026-03-12 Task 8 File Walker

- Added `walker.rs` in `truesight-engine` with an `ignore::WalkBuilder`-based `FileWalker` that respects `.gitignore`, filters supported source extensions, enforces the plan's hardcoded excluded directories, skips binary files by scanning the first 8KB for NUL bytes, and skips files above a configurable size threshold (default 1 MiB).
- `DiscoveredFile` returns metadata only: path, detected `Language`, and file size.
- Added walker tests covering real Rust/TypeScript/C# fixtures plus synthetic cases for `.gitignore`, `node_modules`, binary files, and max-size filtering.

## 2026-03-12 Task 5 Parser Extraction

- Added `truesight-engine/src/parser.rs` with a tree-sitter-backed `CodeParser`, `parse_file()`, and `detect_language()` for Rust, TypeScript, and C#
- Parser walks AST nodes pragmatically in V1: exported/public top-level symbols plus class/impl methods, nested C# enums, and fixture-backed type/property fallbacks mapped into `CodeUnitKind::Constant`
- Parser test coverage is fixture-driven and validates language detection, expected symbol names/kinds, doc extraction, signatures, line ranges, parent linkage, and error-tolerant parsing
- A temporary standalone crate at `/tmp/truesight-parser-check` can compile and run the parser module tests even while unrelated engine modules fail workspace builds

## 2026-03-12 Task 6 Storage Backend

- `truesight-db` now implements the `Storage` trait on `Database` with transaction-wrapped upserts into `code_units` and CRUD for `index_metadata`
- FTS reads join `code_units_fts` back to `code_units`, so trigger-synced lexical search stays scoped to `repo_id + branch`
- Embeddings are stored as little-endian `f32` BLOBs and vector lookup falls back to Rust cosine similarity when `vector_top_k(...)` is unavailable
- Added db-level helpers for `indexed_files`, file-scoped deletes, and embedding-aware writes to support later indexing work without changing the Task 4 schema

## 2026-03-12 Task 7 Embedding Runtime Fix

- `truesight-engine/src/embed.rs` now resolves ONNX Runtime from only `ORT_DYLIB_PATH` or build-time `TRUESIGHT_ORT_DYLIB`, initializes ORT eagerly, and returns a clear error when neither path is available.
- Runtime `/nix/store` scanning was removed from `embed.rs`; build-time selection is now deterministic via `truesight-engine/build.rs` using a validated pinned ORT dylib path in this Nix environment.
- ONNX output extraction now prefers named outputs (`sentence_embedding`, `token_embeddings`, `last_hidden_state`) and otherwise selects a unique tensor by rank and final dimension instead of assuming `outputs[0]`.
- Verified: `nix develop --command cargo test -p truesight-engine -- embed::tests::test_dimension -- --nocapture`, `nix develop --command cargo test -p truesight-engine -- embed -- --nocapture`, and `nix develop --command cargo build --workspace` all pass.

## 2026-03-12 Task 7 - Embedder Runtime Hang

- The remaining hang was in ORT environment/session initialization rather than model download or tokenization; an explicit one-time `ort::init().commit()` before `Session::builder()` removes the deadlock.
- `truesight-engine` now uses non-`load-dynamic` ORT features with `default-features = false` plus explicit `download-binaries`, `copy-dylibs`, `api-24`, and `tls-rustls`, which avoids the prior runtime loading path that hung and also avoids the earlier `libssl.so.3` build-script failure from native TLS.
- The cached all-MiniLM-L6-v2 ONNX artifact still works with the existing mean-pooling implementation, and the real embedding tests now finish in about 5 seconds total instead of hanging for minutes.

## 2026-03-12 Task 7 Embedding Engine

- Removed the blocking `ort::init_from(...).commit()` path; setting `ORT_DYLIB_PATH` before `Session::builder()` avoids the embedder startup hang while keeping ONNX Runtime loading deterministic.
- `build.rs` now supplies a deterministic `TRUESIGHT_ORT_DYLIB` fallback plus Linux runtime rpaths for native dependencies, so the required embed tests and workspace build pass in this Nix environment.
- ONNX output extraction now prefers named 2D sentence outputs (`sentence_embedding`) and falls back to named 3D token outputs (`token_embeddings` / `last_hidden_state`) before using shape-based disambiguation.

## 2026-03-12 Task 7 Runtime Alignment Follow-up

- `truesight-engine/build.rs` is more portable when it selects the highest compatible `/nix/store` `libonnxruntime.so` instead of pinning a single store hash; that keeps runtime loading deterministic without baking in one machine-specific path.
- Matching `ort`'s API level to the available Nix runtime (`api-23` with `onnxruntime-1.23.x`) keeps `load-dynamic` startup fast and preserves the passing MiniLM runtime tests.
- Re-running the exact Wave 2 verification commands after the runtime-selection cleanup still finishes in about 4-5 seconds for embed tests and succeeds for the full workspace build.

## 2026-03-12 Task 9 Indexing Pipeline

- Added `truesight-engine/src/indexer.rs` with an `Indexer` that does full-index orchestration in the intended order: repo context detection -> `FileWalker` discovery -> `CodeParser` extraction -> `Embedder` inference -> storage writes + metadata updates.
- `IndexStorage` extends the shared `Storage` contract just enough for indexing-specific persistence (`store_indexed_units`, `upsert_indexed_file`), and `truesight_db::Database` now plugs into that adapter without moving orchestration into the DB crate.
- Index stats are computed from actual processed files, including per-language file counts, embedded chunk totals, duration, and graceful skip behavior for empty repos, files with no symbols, and per-file parse/embed failures.
- Fixture-backed indexer tests now use a real temp `Database`, so Task 9 verifies stored symbols, stored embeddings, indexed-file bookkeeping, and metadata round-trips instead of only testing in-memory mocks.

## 2026-03-12 Task 12 Incremental Indexing

- Added `truesight-engine/src/incremental.rs` with git-first `detect_changes()` and a SHA-256 file-hash fallback that compares current walker output against `indexed_files` when git metadata is missing or unusable.
- Incremental updates now handle added, modified, and deleted files explicitly, reuse parser/embedder flow per file, delete stale file units before reindex, and prune deleted-file rows from `indexed_files`.
- Verification passed for `nix develop --command cargo test -p truesight-engine -- incremental` and `nix develop --command cargo build --workspace`.

## 2026-03-12 Task 11 Repo Map Generator

- Added `truesight-engine/src/repomap.rs` with `RepoMapGenerator::generate(...)`, implemented as a pure projection over `Storage::get_all_symbols(...)` with no parser/tree-sitter fallback.
- Repo map grouping is deterministic via ordered maps/sets: modules sort by relative directory path, files sort alphabetically within each module, and symbols sort by relative file path then line then name.
- Root-level files map to a `.` module path with the repo directory name as the module name; dependency hints are a stable V1 summary derived from symbol names mentioned in stored signatures/content.
- Added repo-map tests covering rust fixture structure, module ordering, missing docs, empty storage, and deterministic dependency hints from stored symbol data.

## 2026-03-12 Task 10 Search Engine

- Added `truesight-engine/src/search.rs` as the orchestration layer over `Storage` + `Embedder`, with async lexical search, optional semantic search, and graceful degradation when embeddings or vector lookup are unavailable.
- Reciprocal rank fusion is rank-based only (`sum(1 / (rrf_k + rank))`), merged by a stable code-unit key, then normalized to 0-1 before `min_score` filtering and `limit` truncation.
- Final `SearchResult` values now carry stable `match_type` (`fts`, `vector`, `hybrid`) plus symbol-level snippets derived from stored code-unit content.
- `nix develop --command cargo test -p truesight-engine -- search` and `nix develop --command cargo build --workspace` both pass after wiring the new search module.

## 2026-03-12 Task 14 MCP Server

- `truesight/src/mcp.rs` now hosts an rmcp stdio server with `#[tool_router]` + `#[tool_handler]`, exposing exactly `index_repo`, `search_repo`, and `repo_map`.
- MCP handlers delegate through `truesight/src/app.rs` service methods, so indexing, search, freshness checks, and repo-map generation stay in the existing engine/db layers instead of being reimplemented in transport code.
- Returning JSON strings from the rmcp tool methods avoids cross-crate `schemars` incompatibilities between the existing core types and rmcp's schema expectations while keeping the tool output contract faithful.
- Verified with `nix develop --command cargo test -p truesight`, `nix develop --command cargo build -p truesight`, and a bash coprocess smoke test that completes `initialize` plus `tools/list` and reports all three tools.

## 2026-03-12 Task 13 CLI with clap

- Replaced the placeholder `truesight` binary with clap-derive parsing for `mcp`, `index`, and `search` in `truesight/src/main.rs`, `truesight/src/cli.rs`, and `truesight/src/app.rs`.
- The CLI now stays thin: `AppServices` owns DB/embedder setup, uses `truesight-engine::detect_repo_context`, `index_repo`, `IncrementalIndexer`, `SearchEngine`, and `RepoMapGenerator`, and formats stable text output for later CLI integration checks.
- `search` now treats an existing symbol index as sufficient even if branch metadata is missing, which avoids false "not indexed" errors when older/full index state exists without a matching `index_metadata` row.
- Added package-local tests for help parsing, full-index stats output, post-index search output, and the unindexed search error path; `nix develop --command cargo test -p truesight`, `nix develop --command cargo run -p truesight -- --help`, and `nix develop --command cargo run -p truesight -- search --help` all pass.

## 2026-03-12 Task 15 Branch Management + DB Partitioning

- Added a centralized `truesight-engine/src/repo_context.rs` helper so repo-root canonicalization, hashed `repo_id` generation, named-branch detection, and detached-HEAD fallback all resolve through one code path.
- Detached HEAD now partitions indexes under `detached-<full_sha>` instead of ambiguous `HEAD`, while non-git directories fall back to `default`.
- Added root-aware convenience entrypoints in engine search/repomap/incremental flows so future CLI/MCP code can reuse the same branch-aware partition detection without duplicating git logic.
- Added DB coverage for distinct branch listing, branch-scoped search/metadata isolation, and branch cleanup that preserves other partitions.
- Verified: `nix develop --command cargo test -p truesight-engine repo_context::tests`, `nix develop --command cargo test -p truesight-engine -- branch`, `nix develop --command cargo test -p truesight-db`, and `nix develop --command cargo build --workspace`.

## 2026-03-12 Task 14 MCP Server with rmcp

- Replaced the CLI placeholder MCP entrypoint with a real rmcp stdio server in `truesight/src/mcp.rs` and wired `main.rs`/`app.rs` to call it through the `mcp` subcommand.
- Registered exactly three rmcp tools - `index_repo`, `search_repo`, and `repo_map` - and kept handlers thin by delegating all indexing/search/repo-map work to `AppServices`.
- Used rmcp's `Json` structured-output wrapper with MCP-local request/response DTOs so tool schemas stay compatible with rmcp 1.2 without pulling the older `schemars` derives from shared core/app types into the tool layer.
- rmcp 1.2's stdio transport in this workspace uses newline-delimited JSON-RPC over async read/write rather than `Content-Length` framing, so the bash smoke test talks to the server with plain JSON lines.

## 2026-03-12 Task 14 Verification Follow-up

- Current on-disk `truesight/src/mcp.rs` already matches the planned rmcp 1.2 stdio design: `transport::io::stdio`, rmcp-local `JsonSchema` request DTOs, non-exhaustive `ServerInfo::new(...).with_instructions(...)`, and exactly three registered tools.
- The focused router unit test still asserts the exported tool set is exactly `index_repo`, `repo_map`, and `search_repo`, which protects later tasks from accidental MCP surface-area drift.
- A bash coprocess smoke test successfully completes `initialize` and `tools/list`, confirming the stdio path is usable for later MCP protocol coverage.

## 2026-03-12 Task 17 MCP Protocol Tests

- Added `truesight/tests/mcp_protocol.rs` with real spawned-process stdio protocol coverage using the built `truesight mcp` binary, newline-delimited JSON-RPC requests, and deterministic child cleanup in `Drop`.
- The protocol test now asserts the `initialize` contract, verifies `tools/list` exposes exactly `index_repo`, `repo_map`, and `search_repo`, and checks that each listed tool publishes both input and output schemas.
- End-to-end tool coverage now indexes the Rust fixture through `tools/call`, asserts exact JSON-RPC `structuredContent` fields (including branch, repo root, file counts, symbol counts, and language counts), then calls `repo_map` and verifies concrete module/file/symbol data from the spawned server.
- Verification passed with `nix develop --command cargo test -p truesight mcp` and `nix develop --command cargo test --workspace`.

## 2026-03-12 Task 16 End-to-End Integration Tests

- Added `truesight/tests/cli_e2e.rs` with temp-repo integration coverage that drives the real `app::run(...)` CLI entrypoint through clap parsing instead of unit-testing helpers.
- The new tests cover a full index -> search success path against a cloned fixture repo and a branch-isolation flow that indexes `main`, creates a feature branch with a branch-only symbol, reindexes, and verifies the symbol does not bleed back into `main` results.
- Added `truesight/src/lib.rs` so integration tests can reuse the real CLI/app modules without duplicating wiring, and updated `truesight/src/main.rs` to call through that library surface.
- Existing `truesight/src/app.rs` tests now use a shared lock around fixture-backed DB cases so `cargo test -p truesight` stays deterministic after exposing the library target.
- Verified: `nix develop --command cargo test -p truesight` and `nix develop --command cargo test --workspace` both pass with the new coverage.

## 2026-03-12 Task 18 Degraded + Edge-Case Tests

- Added search-engine coverage for degraded lexical fallback when no embedder is configured and when vector lookup fails after query embedding succeeds.
- Added indexer coverage for malformed Rust source that still yields partial symbols plus a Unicode-path case that verifies stored unit paths survive non-ASCII filenames.
- Added CLI/app coverage for two user-facing edge cases: searching an indexed empty repo returns `no_results: true`, and an empty search query fails as an error instead of crashing.
- Added `truesight/src/lib.rs` so the `truesight` package exposes its existing app/cli/mcp modules to package tests and integration tests during workspace verification.

## 2026-03-12 Final Architecture Compliance Fix

- Moved indexing-specific persistence contracts into `truesight-core`, so `truesight-engine` now talks only to core storage types/methods and no longer imports `truesight-db`.
- `truesight-db::Database` now implements the shared indexing/incremental storage operations directly, while engine tests use in-crate storage doubles instead of a DB dependency.
- Reduced `truesight-engine`'s public exports to the symbols consumed by `truesight`, which removes the extra surface area flagged by final review.

## 2026-03-12 Final Scope Fidelity Cleanup

- Restored `truesight-core::Storage` to the shared contract surface and moved indexing-only operations behind narrower `IndexStorage` and `IncrementalStorage` traits.
- Removed the `truesight` package library target and rewrote `truesight/tests/cli_e2e.rs` to exercise the real `truesight` binary via `CARGO_BIN_EXE_truesight`, which preserves behavior while dropping unnecessary public module exports.
- Narrowed crate-local app/indexer abstractions: `truesight/src/app.rs` now keeps app response types/services crate-scoped, `truesight/src/mcp.rs` owns the internal app trait used for test doubles, and `truesight-engine/src/indexer.rs` no longer exposes its single-impl helper traits publicly.

## 2026-03-12 Fixture Repo De-embedding

- Added a shared test helper at `tests/test_support/git_fixture.rs` that copies plain fixture files into a temp directory, runs non-interactive `git init -b main`, and commits there so git-dependent tests no longer rely on nested fixture repos.
- Updated fixture-backed git tests in `truesight/tests/cli_e2e.rs`, `truesight/src/app.rs`, and `truesight-engine/src/indexer.rs` to use temp repos, while parser/repomap fixture readers continue to use the plain tracked fixture directories directly.
- Removed embedded `.git` directories from `tests/fixtures/rust-fixture`, `tests/fixtures/ts-fixture`, and `tests/fixtures/csharp-fixture`; `nix develop --command cargo test --workspace` still passes after removal.

## 2026-03-12 Task 19 ONNX Runtime Devshell Alignment

- `flake.nix` now adds `pkgs.onnxruntime` to the devshell and exports deterministic Linux runtime env vars: `ORT_DYLIB_PATH=${pkgs.onnxruntime}/lib/libonnxruntime.so` plus an `LD_LIBRARY_PATH` built from ONNX Runtime, OpenSSL, and the GCC runtime.
- `truesight-engine/build.rs` no longer scans `/nix/store`; it now validates only the explicit `ORT_DYLIB_PATH`, forwards it into `TRUESIGHT_ORT_DYLIB`, and derives any linker rpaths from the existing `LD_LIBRARY_PATH`.
- `truesight/build.rs` also stopped probing `/nix/store` and now uses the explicit `LD_LIBRARY_PATH` entries for Linux rpaths instead.
- `truesight-engine/src/embed.rs` still prefers runtime `ORT_DYLIB_PATH` first, keeps the build-time `TRUESIGHT_ORT_DYLIB` fallback second, and now points users toward `nix develop` in the missing-runtime error message.
- Verified with `nix develop --command cargo test -p truesight-engine -- embed`, `nix develop --command cargo test --workspace`, and `nix develop --command cargo clippy --workspace --all-targets -- -D warnings`.
