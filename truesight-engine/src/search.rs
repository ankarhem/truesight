use std::path::Path;

use std::cmp::Ordering;

use tracing::warn;
use truesight_core::{
    CodeUnit, Embedder, MatchType, RankedResult, Result, SearchConfig, SearchResult, Storage,
};

use crate::repo_context::detect_repo_context;

const SNIPPET_CONTEXT_LINES: usize = 3;

pub struct SearchEngine<'a> {
    storage: &'a dyn Storage,
    embedder: Option<&'a dyn Embedder>,
}

impl<'a> SearchEngine<'a> {
    pub fn new(storage: &'a dyn Storage, embedder: Option<&'a dyn Embedder>) -> Self {
        Self { storage, embedder }
    }

    pub async fn search(
        &self,
        query: &str,
        repo_id: &str,
        branch: &str,
        config: &SearchConfig,
    ) -> Result<Vec<SearchResult>> {
        search(query, repo_id, branch, self.storage, self.embedder, config).await
    }

    pub async fn search_repo(
        &self,
        query: &str,
        root: &Path,
        config: &SearchConfig,
    ) -> Result<Vec<SearchResult>> {
        search_repo(query, root, self.storage, self.embedder, config).await
    }
}

pub async fn search_repo(
    query: &str,
    root: &Path,
    storage: &dyn Storage,
    embedder: Option<&dyn Embedder>,
    config: &SearchConfig,
) -> Result<Vec<SearchResult>> {
    let context = detect_repo_context(root)?;
    search(
        query,
        &context.repo_id,
        &context.branch,
        storage,
        embedder,
        config,
    )
    .await
}

pub async fn search(
    query: &str,
    repo_id: &str,
    branch: &str,
    storage: &dyn Storage,
    embedder: Option<&dyn Embedder>,
    config: &SearchConfig,
) -> Result<Vec<SearchResult>> {
    if config.use_fts && config.use_vector {
        return search_hybrid_or_lexical(query, repo_id, branch, storage, embedder, config).await;
    }

    if config.use_fts {
        return search_lexical(query, repo_id, branch, storage, config).await;
    }

    if config.use_vector {
        return search_vector_only(query, repo_id, branch, storage, embedder, config).await;
    }

    Ok(Vec::new())
}

async fn search_hybrid_or_lexical(
    query: &str,
    repo_id: &str,
    branch: &str,
    storage: &dyn Storage,
    embedder: Option<&dyn Embedder>,
    config: &SearchConfig,
) -> Result<Vec<SearchResult>> {
    let Some(embedder) = embedder else {
        warn!(
            repo_id,
            branch, query, "no embedder configured; degrading search"
        );
        return search_lexical(query, repo_id, branch, storage, config).await;
    };

    let Some(embedding) = query_embedding(query, repo_id, branch, embedder) else {
        return search_lexical(query, repo_id, branch, storage, config).await;
    };

    match storage
        .search_hybrid(
            repo_id,
            branch,
            query,
            &embedding,
            config.limit,
            config.rrf_k,
        )
        .await
    {
        Ok(results) => Ok(ranked_results_to_search_results(results, config)),
        Err(error) => {
            warn!(
                repo_id,
                branch, query, "hybrid search degraded to lexical-only mode: {error}"
            );
            search_lexical(query, repo_id, branch, storage, config).await
        }
    }
}

async fn search_lexical(
    query: &str,
    repo_id: &str,
    branch: &str,
    storage: &dyn Storage,
    config: &SearchConfig,
) -> Result<Vec<SearchResult>> {
    let fts_results = storage
        .search_fts(repo_id, branch, query, pre_fusion_limit(config.limit))
        .await?;
    Ok(ranked_results_to_search_results(fts_results, config))
}

async fn search_vector_only(
    query: &str,
    repo_id: &str,
    branch: &str,
    storage: &dyn Storage,
    embedder: Option<&dyn Embedder>,
    config: &SearchConfig,
) -> Result<Vec<SearchResult>> {
    let Some(embedder) = embedder else {
        warn!(
            repo_id,
            branch, query, "no embedder configured; degrading search"
        );
        return Ok(Vec::new());
    };

    let Some(embedding) = query_embedding(query, repo_id, branch, embedder) else {
        return Ok(Vec::new());
    };

    let vector_results = storage
        .search_vector(repo_id, branch, &embedding, pre_fusion_limit(config.limit))
        .await
        .unwrap_or_else(|error| {
            warn!(
                repo_id,
                branch, query, "vector search degraded to empty mode: {error}"
            );
            Vec::new()
        });

    Ok(ranked_results_to_search_results(vector_results, config))
}

fn query_embedding(
    query: &str,
    repo_id: &str,
    branch: &str,
    embedder: &dyn Embedder,
) -> Option<Vec<f32>> {
    match embedder.embed(query) {
        Ok(embedding) => Some(embedding),
        Err(error) => {
            warn!(
                repo_id,
                branch, query, "query embedding unavailable; degrading search: {error}"
            );
            None
        }
    }
}

fn ranked_results_to_search_results(
    ranked_results: Vec<RankedResult>,
    config: &SearchConfig,
) -> Vec<SearchResult> {
    if ranked_results.is_empty() {
        return Vec::new();
    }

    let max_score = ranked_results
        .iter()
        .map(|result| result.combined_score)
        .fold(0.0_f32, f32::max);

    let mut results = ranked_results
        .into_iter()
        .filter_map(|result| {
            let normalized_score = if max_score > 0.0 {
                result.combined_score / max_score
            } else {
                0.0
            };

            let snippet = snippet_from_content(&result.unit);

            (normalized_score >= config.min_score).then_some(SearchResult {
                kind: result.unit.kind,
                name: result.unit.name,
                path: result.unit.file_path,
                line: result.unit.line_start,
                signature: result.unit.signature,
                doc: result.unit.doc,
                snippet,
                score: normalized_score,
                match_type: result.match_type,
            })
        })
        .collect::<Vec<_>>();

    results.sort_by(compare_search_results);
    results.truncate(config.limit);
    results
}

fn pre_fusion_limit(limit: usize) -> usize {
    limit.max(1).saturating_mul(3)
}

fn compare_search_results(left: &SearchResult, right: &SearchResult) -> Ordering {
    right
        .score
        .partial_cmp(&left.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| match_type_rank(right.match_type).cmp(&match_type_rank(left.match_type)))
        .then_with(|| left.line.cmp(&right.line))
        .then_with(|| left.path.cmp(&right.path))
        .then_with(|| left.name.cmp(&right.name))
}

fn match_type_rank(match_type: MatchType) -> u8 {
    match match_type {
        MatchType::Hybrid => 2,
        MatchType::Fts | MatchType::Vector => 1,
    }
}

fn snippet_from_content(unit: &CodeUnit) -> String {
    let lines = unit.content.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return unit.content.clone();
    }

    let end = (SNIPPET_CONTEXT_LINES * 2 + 1).min(lines.len());
    lines[..end].join("\n")
}

#[cfg(test)]
#[path = "search_tests.rs"]
mod tests;
