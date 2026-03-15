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
    let pre_fusion_limit = pre_fusion_limit(config.limit);

    if config.use_fts && config.use_vector {
        let Some(embedder) = embedder else {
            warn!(
                repo_id,
                branch, query, "no embedder configured; degrading search"
            );
            let fts_results = storage
                .search_fts(repo_id, branch, query, pre_fusion_limit)
                .await?;
            return Ok(ranked_results_to_search_results(fts_results, config));
        };

        let embedding = match embedder.embed(query) {
            Ok(embedding) => embedding,
            Err(error) => {
                warn!(
                    repo_id,
                    branch, query, "query embedding unavailable; degrading search: {error}"
                );
                let fts_results = storage
                    .search_fts(repo_id, branch, query, pre_fusion_limit)
                    .await?;
                return Ok(ranked_results_to_search_results(fts_results, config));
            }
        };

        return match storage
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
                let fts_results = storage
                    .search_fts(repo_id, branch, query, pre_fusion_limit)
                    .await?;
                Ok(ranked_results_to_search_results(fts_results, config))
            }
        };
    }

    if config.use_fts {
        let fts_results = storage
            .search_fts(repo_id, branch, query, pre_fusion_limit)
            .await?;
        return Ok(ranked_results_to_search_results(fts_results, config));
    }

    if config.use_vector {
        let vector_results = match embedder {
            Some(embedder) => match embedder.embed(query) {
                Ok(embedding) => storage
                    .search_vector(repo_id, branch, &embedding, pre_fusion_limit)
                    .await
                    .unwrap_or_else(|error| {
                        warn!(
                            repo_id,
                            branch, query, "vector search degraded to empty mode: {error}"
                        );
                        Vec::new()
                    }),
                Err(error) => {
                    warn!(
                        repo_id,
                        branch, query, "query embedding unavailable; degrading search: {error}"
                    );
                    Vec::new()
                }
            },
            None => {
                warn!(
                    repo_id,
                    branch, query, "no embedder configured; degrading search"
                );
                Vec::new()
            }
        };
        return Ok(ranked_results_to_search_results(vector_results, config));
    }

    Ok(Vec::new())
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
mod tests {
    use std::path::PathBuf;

    use truesight_core::{
        CodeUnitKind, Language, MatchType, MockEmbedder, MockStorage, RankedResult, SearchConfig,
        TruesightError,
    };

    use super::{SearchEngine, search};

    fn expect_fts_results(storage: &mut MockStorage, results: Vec<RankedResult>) {
        storage
            .expect_search_fts()
            .times(1)
            .returning(move |_, _, _, limit| Ok(results.iter().take(limit).cloned().collect()));
    }

    fn expect_vector_results(storage: &mut MockStorage, results: Vec<RankedResult>) {
        storage
            .expect_search_vector()
            .times(1)
            .returning(move |_, _, _, limit| Ok(results.iter().take(limit).cloned().collect()));
    }

    fn expect_hybrid_results(storage: &mut MockStorage, results: Vec<RankedResult>) {
        storage
            .expect_search_hybrid()
            .times(1)
            .returning(move |_, _, _, _, limit, _| {
                Ok(results.iter().take(limit).cloned().collect())
            });
    }

    fn expect_hybrid_error(storage: &mut MockStorage, message: &str) {
        let message = message.to_string();
        storage
            .expect_search_hybrid()
            .times(1)
            .returning(move |_, _, _, _, _, _| Err(TruesightError::Database(message.clone())));
    }

    fn expect_embedder_success(embedder: &mut MockEmbedder, embedding: Vec<f32>) {
        embedder
            .expect_embed()
            .times(1)
            .returning(move |_| Ok(embedding.clone()));
    }

    fn expect_embedder_error(embedder: &mut MockEmbedder, message: &str) {
        let message = message.to_string();
        embedder
            .expect_embed()
            .times(1)
            .returning(move |_| Err(TruesightError::Embedding(message.clone())));
    }

    #[tokio::test]
    async fn lexical_only_search_returns_ranked_fts_results() {
        let mut storage = MockStorage::new();
        expect_fts_results(
            &mut storage,
            vec![fts_result(
                sample_unit("AuthService", "src/auth.rs", 8),
                0.8,
            )],
        );
        let config = SearchConfig {
            limit: 5,
            rrf_k: 60,
            use_fts: true,
            use_vector: false,
            min_score: 0.0,
        };

        let results = search("auth struct", "/repo", "main", &storage, None, &config)
            .await
            .expect("fts-only search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "AuthService");
        assert_eq!(results[0].match_type, MatchType::Fts);
        assert!((results[0].score - 1.0).abs() < f32::EPSILON);
        assert!(results[0].snippet.contains("pub struct AuthService"));
    }

    #[tokio::test]
    async fn vector_assisted_search_returns_vector_results_when_fts_is_disabled() {
        let mut storage = MockStorage::new();
        expect_vector_results(
            &mut storage,
            vec![vector_result(
                sample_unit("semantic_retry", "src/retry.rs", 14),
                0.93,
            )],
        );
        let mut embedder = MockEmbedder::new();
        expect_embedder_success(&mut embedder, vec![0.4, 0.6, 0.2]);
        let config = SearchConfig {
            limit: 5,
            rrf_k: 60,
            use_fts: false,
            use_vector: true,
            min_score: 0.0,
        };

        let results = SearchEngine::new(&storage, Some(&embedder))
            .search("retry failed payments", "/repo", "main", &config)
            .await
            .expect("vector-only search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "semantic_retry");
        assert_eq!(results[0].match_type, MatchType::Vector);
        assert!((results[0].score - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn hybrid_search_uses_db_fused_results_on_happy_path() {
        let mut storage = MockStorage::new();
        expect_hybrid_results(
            &mut storage,
            vec![RankedResult {
                unit: sample_unit("retry_job", "src/retry.rs", 22),
                fts_score: Some(0.4),
                vector_score: Some(0.5),
                combined_score: 0.9,
                match_type: MatchType::Hybrid,
            }],
        );
        let mut embedder = MockEmbedder::new();
        expect_embedder_success(&mut embedder, vec![0.2, 0.4, 0.6]);

        let results = SearchEngine::new(&storage, Some(&embedder))
            .search(
                "retry failed payments",
                "/repo",
                "main",
                &SearchConfig::default(),
            )
            .await
            .expect("hybrid search should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "retry_job");
        assert_eq!(results[0].match_type, MatchType::Hybrid);
        assert!((results[0].score - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn hybrid_search_preserves_storage_ordering_and_match_types() {
        let mut storage = MockStorage::new();
        expect_hybrid_results(
            &mut storage,
            vec![
                RankedResult {
                    unit: sample_unit("retry_job", "src/retry.rs", 22),
                    fts_score: Some(0.4),
                    vector_score: Some(0.5),
                    combined_score: 0.9,
                    match_type: MatchType::Hybrid,
                },
                RankedResult {
                    unit: sample_unit("AuthService", "src/auth.rs", 8),
                    fts_score: Some(0.6),
                    vector_score: None,
                    combined_score: 0.5,
                    match_type: MatchType::Fts,
                },
                RankedResult {
                    unit: sample_unit("cache_layer", "src/cache.rs", 40),
                    fts_score: None,
                    vector_score: Some(0.4),
                    combined_score: 0.4,
                    match_type: MatchType::Vector,
                },
            ],
        );
        let mut embedder = MockEmbedder::new();
        expect_embedder_success(&mut embedder, vec![0.2, 0.4, 0.6]);

        let results = SearchEngine::new(&storage, Some(&embedder))
            .search("retry auth", "/repo", "main", &SearchConfig::default())
            .await
            .expect("hybrid search should succeed");

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].name, "retry_job");
        assert_eq!(results[0].match_type, MatchType::Hybrid);
        assert_eq!(results[1].name, "AuthService");
        assert_eq!(results[1].match_type, MatchType::Fts);
        assert_eq!(results[2].name, "cache_layer");
        assert_eq!(results[2].match_type, MatchType::Vector);
    }

    #[tokio::test]
    async fn hybrid_search_applies_min_score_and_limit_from_storage_scores() {
        let mut storage = MockStorage::new();
        expect_hybrid_results(
            &mut storage,
            vec![
                RankedResult {
                    unit: sample_unit("first", "src/first.rs", 1),
                    fts_score: Some(0.5),
                    vector_score: Some(0.5),
                    combined_score: 0.8,
                    match_type: MatchType::Hybrid,
                },
                RankedResult {
                    unit: sample_unit("second", "src/second.rs", 2),
                    fts_score: Some(0.4),
                    vector_score: None,
                    combined_score: 0.7,
                    match_type: MatchType::Fts,
                },
            ],
        );
        let mut embedder = MockEmbedder::new();
        expect_embedder_success(&mut embedder, vec![0.2, 0.4, 0.6]);
        let config = SearchConfig {
            limit: 1,
            rrf_k: 60,
            use_fts: true,
            use_vector: true,
            min_score: 0.99,
        };

        let results = SearchEngine::new(&storage, Some(&embedder))
            .search("first", "/repo", "main", &config)
            .await
            .expect("hybrid search should honor score filtering");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "first");
        assert!((results[0].score - 1.0).abs() < f32::EPSILON);
    }

    #[tokio::test]
    async fn degraded_mode_returns_fts_results_when_embedder_fails() {
        let mut storage = MockStorage::new();
        expect_fts_results(
            &mut storage,
            vec![fts_result(sample_unit("User", "src/user.rs", 3), 0.88)],
        );
        let mut failing_embedder = MockEmbedder::new();
        expect_embedder_error(&mut failing_embedder, "model unavailable");
        let config = SearchConfig::default();

        let results = search(
            "user model",
            "/repo",
            "main",
            &storage,
            Some(&failing_embedder),
            &config,
        )
        .await
        .expect("degraded search should still succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "User");
        assert_eq!(results[0].match_type, MatchType::Fts);
    }

    #[tokio::test]
    async fn degraded_mode_returns_fts_results_when_no_embedder_is_configured() {
        let mut storage = MockStorage::new();
        expect_fts_results(
            &mut storage,
            vec![fts_result(sample_unit("User", "src/user.rs", 3), 0.88)],
        );

        let results = search(
            "user model",
            "/repo",
            "main",
            &storage,
            None,
            &SearchConfig::default(),
        )
        .await
        .expect("search without embedder should degrade to lexical-only mode");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "User");
        assert_eq!(results[0].match_type, MatchType::Fts);
    }

    #[tokio::test]
    async fn degraded_mode_returns_fts_results_when_vector_lookup_fails() {
        let mut storage = MockStorage::new();
        expect_hybrid_error(&mut storage, "vector index unavailable");
        expect_fts_results(
            &mut storage,
            vec![fts_result(
                sample_unit("AuthService", "src/auth.rs", 8),
                0.8,
            )],
        );
        let mut embedder = MockEmbedder::new();
        expect_embedder_success(&mut embedder, vec![0.4, 0.6, 0.2]);

        let results = search(
            "authentication",
            "/repo",
            "main",
            &storage,
            Some(&embedder),
            &SearchConfig::default(),
        )
        .await
        .expect("vector backend failures should degrade to lexical-only mode");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "AuthService");
        assert_eq!(results[0].match_type, MatchType::Fts);
    }

    #[tokio::test]
    async fn degraded_mode_returns_vector_results_when_fts_is_empty() {
        let mut storage = MockStorage::new();
        expect_hybrid_results(
            &mut storage,
            vec![vector_result(
                sample_unit("RetryPolicy", "src/retry.rs", 19),
                0.97,
            )],
        );
        let mut embedder = MockEmbedder::new();
        expect_embedder_success(&mut embedder, vec![0.1, 0.2, 0.3]);
        let config = SearchConfig::default();

        let results = search(
            "retry backoff",
            "/repo",
            "main",
            &storage,
            Some(&embedder),
            &config,
        )
        .await
        .expect("vector fallback should succeed");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "RetryPolicy");
        assert_eq!(results[0].match_type, MatchType::Vector);
    }

    fn sample_unit(name: &str, path: &str, line: u32) -> truesight_core::CodeUnit {
        truesight_core::CodeUnit {
            name: name.to_string(),
            kind: if name.ends_with("Service") || name == "User" || name == "RetryPolicy" {
                CodeUnitKind::Struct
            } else {
                CodeUnitKind::Function
            },
            signature: match name {
                "AuthService" => "pub struct AuthService".to_string(),
                "User" => "pub struct User".to_string(),
                "RetryPolicy" => "pub struct RetryPolicy".to_string(),
                other => format!("pub fn {other}(input: &str) -> bool"),
            },
            doc: Some(format!("Documentation for {name}")),
            file_path: PathBuf::from(path),
            line_start: line,
            line_end: line + 4,
            content: match name {
                "AuthService" => {
                    "pub struct AuthService {\n    pub enabled: bool,\n}\n\nimpl AuthService {\n    pub fn is_enabled(&self) -> bool {\n        self.enabled\n    }\n}".to_string()
                }
                "User" => "pub struct User {\n    pub id: u64,\n    pub name: String,\n}".to_string(),
                "RetryPolicy" => {
                    "pub struct RetryPolicy {\n    pub max_retries: u32,\n    pub backoff_ms: u64,\n}".to_string()
                }
                other => format!(
                    "pub fn {other}(input: &str) -> bool {{\n    input.contains(\"{other}\")\n}}"
                ),
            },
            parent: None,
            language: Language::Rust,
        }
    }

    fn fts_result(unit: truesight_core::CodeUnit, score: f32) -> RankedResult {
        RankedResult {
            unit,
            fts_score: Some(score),
            vector_score: None,
            combined_score: score,
            match_type: MatchType::Fts,
        }
    }

    fn vector_result(unit: truesight_core::CodeUnit, score: f32) -> RankedResult {
        RankedResult {
            unit,
            fts_score: None,
            vector_score: Some(score),
            combined_score: score,
            match_type: MatchType::Vector,
        }
    }
}
