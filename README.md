# Truesight

Code intelligence tool that indexes repositories and provides hybrid search (lexical + semantic) over code symbols. Exposes functionality as both a CLI and an [MCP](https://modelcontextprotocol.io/) server.

## Features

- **Tree-sitter parsing** — Extracts functions, structs, enums, traits, classes, interfaces, and more from Rust, TypeScript, and C#.
- **Hybrid search** — Combines BM25 full-text search (via SQLite FTS5) with vector similarity search (via ONNX Runtime embeddings), fused using Reciprocal Rank Fusion (RRF).
- **Incremental indexing** — Tracks file hashes to only re-index changed files.
- **Repo map** — Generates a structured map of modules, files, symbols, and dependency hints.
- **MCP server** — All functionality available as MCP tools over stdio for integration with AI assistants.

## Usage

### CLI

```bash
# Index a repository
truesight index /path/to/repo
truesight index /path/to/repo --full  # force full reindex

# Search for code symbols
truesight search "authentication middleware" --repo /path/to/repo --limit 20

# Generate a repository map
truesight repo-map /path/to/repo

# Start the MCP server (stdio transport)
truesight mcp
```

### MCP Tools

When running as an MCP server (`truesight mcp`), three tools are exposed:

| Tool | Description |
|------|-------------|
| `search_repo` | Default first step for exploratory codebase work when the exact file or symbol is unknown; ranked lexical + semantic search |
| `repo_map` | Use after search when you need module boundaries, key symbols, and dependency context; pass `filter` to focus on a repo-relative directory or file |
| `index_repo` | Refresh or repair the index when search results are missing, stale, or after major repository changes |

Recommended agent workflow:

1. Start with `search_repo` when the location is unknown or you want likely implementations ranked for you.
2. Use `repo_map` once search has narrowed the area and you want structural context; add `filter` when you already know the relevant directory or file.
3. Use `grep` alongside Truesight when you need exact strings, regex matches, or exhaustive literal confirmation.
4. Use `index_repo` only when the repository changed substantially or search data needs a refresh.

## Architecture

```
truesight/              CLI + MCP server
truesight-core/         Shared types, traits, and error definitions
truesight-engine/       Tree-sitter parsing, ONNX embedding, repo map generation
truesight-db/           libSQL/SQLite storage with FTS5 and vector search
```

### How search works

1. **Parse** — Tree-sitter grammars extract code units (functions, structs, etc.) with signatures, docs, and source.
2. **Embed** — Code units are embedded using a tokenizer + ONNX model for vector representations.
3. **Store** — Code units, FTS5 tokens, and embeddings are persisted in a libSQL database.
4. **Search** — Queries run both FTS5 (BM25 scoring) and vector similarity in parallel, then results are fused with RRF.

## Supported Languages

- Rust
- TypeScript / TSX
- C#

## Building

### With Nix (recommended)

```bash
nix build
./result/bin/truesight --help
```

### With Cargo

```bash
cargo build --release
```

## Install

### With Cargo

```bash
cargo install --path truesight
```

After installation, the `truesight` binary will be available on your `PATH`.

> **Note:** The `ort` crate requires ONNX Runtime. When building with cargo outside of Nix, it will download the binaries automatically.

## Development

This project uses a Nix flake for development:

```bash
# Enter the dev shell (requires Nix with flakes)
nix develop

# Or with direnv
direnv allow
```

The dev shell provides the Rust toolchain (via fenix), rust-analyzer, cargo-watch, cargo-edit, cargo-deny, and pre-commit hooks with treefmt (nixfmt + rustfmt).
