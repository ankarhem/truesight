use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "truesight", about = "Code intelligence tool")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Debug, Subcommand)]
pub enum Commands {
    Mcp,
    Index(IndexArgs),
    Search(SearchArgs),
    RepoMap(RepoMapArgs),
}

#[derive(Debug, Clone, Args)]
pub struct IndexArgs {
    pub path: PathBuf,
    #[arg(long)]
    pub full: bool,
}

#[derive(Debug, Clone, Args)]
pub struct SearchArgs {
    pub query: String,
    #[arg(long, default_value = ".")]
    pub repo: PathBuf,
    #[arg(long, default_value_t = 10)]
    pub limit: usize,
}

#[derive(Debug, Clone, Args)]
pub struct RepoMapArgs {
    #[arg(default_value = ".")]
    pub path: PathBuf,
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::Cli;

    #[test]
    fn help_lists_mcp_index_and_search_subcommands() {
        let mut help = Vec::new();
        Cli::command()
            .write_long_help(&mut help)
            .expect("help should render");
        let text = String::from_utf8(help).expect("help should be utf-8");

        assert!(text.contains("mcp"), "help should mention mcp subcommand");
        assert!(
            text.contains("index"),
            "help should mention index subcommand"
        );
        assert!(
            text.contains("search"),
            "help should mention search subcommand"
        );
    }

    #[test]
    fn search_subcommand_parses_repo_and_limit_arguments() {
        let cli = Cli::try_parse_from([
            "truesight",
            "search",
            "validate email",
            "--repo",
            "tests/fixtures/rust-fixture",
            "--limit",
            "5",
        ])
        .expect("search arguments should parse");

        match cli.command {
            super::Commands::Search(args) => {
                assert_eq!(args.query, "validate email");
                assert_eq!(
                    args.repo,
                    std::path::PathBuf::from("tests/fixtures/rust-fixture")
                );
                assert_eq!(args.limit, 5);
            }
            command => panic!("expected search command, got {command:?}"),
        }
    }
}
