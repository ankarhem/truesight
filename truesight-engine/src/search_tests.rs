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
        .returning(move |_, _, _, _, limit, _| Ok(results.iter().take(limit).cloned().collect()));
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
