use super::*;

#[tokio::test]
async fn search_fts_returns_ranked_match_from_fts_triggers() {
    let (_temp_dir, database) = open_temp_database().await;
    let alpha = sample_unit(
        "alpha_handler",
        "src/alpha.rs",
        10,
        "pub fn alpha_handler() -> bool { true }",
    );
    let target = sample_unit(
        "token_refresh",
        "src/token.rs",
        20,
        "pub fn token_refresh() -> bool { validate_token_refresh() }",
    );
    let gamma = sample_unit(
        "gamma_handler",
        "src/gamma.rs",
        30,
        "pub fn gamma_handler() -> bool { false }",
    );

    database
        .store_code_units(REPO_ID, BRANCH, &[alpha, target.clone(), gamma])
        .await
        .unwrap();

    let results = database
        .search_fts(REPO_ID, BRANCH, "token_refresh", 5)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].unit.name, target.name);
    assert_eq!(results[0].unit.file_path, target.file_path);
    assert_eq!(results[0].match_type, MatchType::Fts);
    assert!(results[0].fts_score.unwrap() > 0.0);

    let empty = database
        .search_fts(REPO_ID, BRANCH, "missing_symbol", 5)
        .await
        .unwrap();
    assert!(empty.is_empty());
}

#[tokio::test]
async fn search_fts_weights_name_and_signature_above_content_only_matches() {
    let (_temp_dir, database) = open_temp_database().await;
    let content_only = sample_unit(
        "helper",
        "src/helper.rs",
        10,
        "pub fn helper() -> bool { let note = \"token_refresh token_refresh\"; true }",
    );
    let exact_symbol = sample_unit(
        "token_refresh",
        "src/token.rs",
        20,
        "pub fn token_refresh() -> bool { true }",
    );

    database
        .store_code_units(REPO_ID, BRANCH, &[content_only, exact_symbol])
        .await
        .unwrap();

    let results = database
        .search_fts(REPO_ID, BRANCH, "token_refresh", 2)
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].unit.name, "token_refresh");
    assert_eq!(results[1].unit.name, "helper");
}

#[test]
fn sanitize_fts_query_quotes_each_token() {
    assert_eq!(
        sanitize_fts_query("hello world"),
        Some(r#""hello" "world""#.to_string())
    );
}

#[test]
fn sanitize_fts_query_escapes_internal_double_quotes() {
    assert_eq!(
        sanitize_fts_query(r#"say "hello""#),
        Some("\"say\" \"\"\"hello\"\"\"".to_string())
    );
}

#[test]
fn sanitize_fts_query_handles_question_mark() {
    assert_eq!(
        sanitize_fts_query("what languages are supported?"),
        Some(r#""what" "languages" "are" "supported?""#.to_string())
    );
}

#[test]
fn sanitize_fts_query_handles_fts_operators() {
    assert_eq!(
        sanitize_fts_query("foo* (bar) -baz"),
        Some(r#""foo*" "(bar)" "-baz""#.to_string())
    );
}

#[test]
fn sanitize_fts_query_returns_none_for_empty_input() {
    assert_eq!(sanitize_fts_query(""), None);
    assert_eq!(sanitize_fts_query("   "), None);
}

#[tokio::test]
async fn search_fts_handles_special_characters_without_error() {
    let (_temp_dir, database) = open_temp_database().await;
    let unit = sample_unit(
        "supported_languages",
        "src/lang.rs",
        10,
        "pub fn supported_languages() -> Vec<String> { vec![] }",
    );

    database
        .store_code_units(REPO_ID, BRANCH, &[unit])
        .await
        .unwrap();

    let results = database
        .search_fts(REPO_ID, BRANCH, "supported?", 5)
        .await
        .unwrap();
    assert!(!results.is_empty());
    assert_eq!(results[0].unit.name, "supported_languages");

    let results = database
        .search_fts(REPO_ID, BRANCH, r#""supported""#, 5)
        .await
        .unwrap();
    assert!(!results.is_empty());

    let results = database
        .search_fts(REPO_ID, BRANCH, "languages (supported)", 5)
        .await
        .unwrap();
    assert!(!results.is_empty());

    let results = database.search_fts(REPO_ID, BRANCH, "", 5).await.unwrap();
    assert!(results.is_empty());

    let results = database
        .search_fts(REPO_ID, BRANCH, "   ", 5)
        .await
        .unwrap();
    assert!(results.is_empty());
}

#[tokio::test]
async fn search_vector_uses_blob_embeddings_and_returns_nearest_neighbor() {
    let (_temp_dir, database) = open_temp_database().await;
    let alpha = IndexedCodeUnit {
        unit: sample_unit(
            "alpha_handler",
            "src/alpha.rs",
            10,
            "pub fn alpha_handler() -> bool { true }",
        ),
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        file_hash: None,
    };
    let beta = IndexedCodeUnit {
        unit: sample_unit(
            "beta_handler",
            "src/beta.rs",
            20,
            "pub fn beta_handler() -> bool { false }",
        ),
        embedding: Some(vec![0.0, 1.0, 0.0, 0.0]),
        file_hash: None,
    };
    let gamma = IndexedCodeUnit {
        unit: sample_unit(
            "gamma_handler",
            "src/gamma.rs",
            30,
            "pub fn gamma_handler() -> bool { false }",
        ),
        embedding: Some(vec![0.8, 0.2, 0.0, 0.0]),
        file_hash: None,
    };

    database
        .store_indexed_units(REPO_ID, BRANCH, &[alpha, beta, gamma])
        .await
        .unwrap();

    let blob = stored_embedding_blob(&database, "alpha_handler").await;
    assert_eq!(blob.len(), 16);
    assert_eq!(decode_embedding(&blob).unwrap(), vec![1.0, 0.0, 0.0, 0.0]);

    let results = database
        .search_vector(REPO_ID, BRANCH, &[1.0, 0.0, 0.0, 0.0], 3)
        .await
        .unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].unit.name, "alpha_handler");
    assert!(results[0].vector_score.unwrap() > results[1].vector_score.unwrap());
    assert_eq!(results[0].match_type, MatchType::Vector);

    let empty = database
        .search_vector(REPO_ID, BRANCH, &[0.0, 0.0, 1.0, 0.0], 0)
        .await
        .unwrap();
    assert!(empty.is_empty());
}

#[tokio::test]
async fn search_hybrid_fuses_lexical_and_vector_hits() {
    let (_temp_dir, database) = open_temp_database().await;
    let hybrid = IndexedCodeUnit {
        unit: sample_unit(
            "authenticate_user",
            "src/auth.rs",
            10,
            "pub fn authenticate_user() -> bool { true }",
        ),
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        file_hash: None,
    };
    let lexical_only = IndexedCodeUnit {
        unit: sample_unit(
            "user_validator",
            "src/user.rs",
            20,
            "pub fn user_validator() -> bool { let label = \"authenticate user\"; true }",
        ),
        embedding: Some(vec![0.0, 1.0, 0.0, 0.0]),
        file_hash: None,
    };
    let vector_only = IndexedCodeUnit {
        unit: sample_unit(
            "semantic_auth",
            "src/semantic.rs",
            30,
            "pub fn semantic_auth() -> bool { false }",
        ),
        embedding: Some(vec![0.95, 0.05, 0.0, 0.0]),
        file_hash: None,
    };

    database
        .store_indexed_units(REPO_ID, BRANCH, &[hybrid, lexical_only, vector_only])
        .await
        .unwrap();

    let results = database
        .search_hybrid(
            REPO_ID,
            BRANCH,
            "authenticate user",
            &[1.0, 0.0, 0.0, 0.0],
            3,
            60,
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].unit.name, "authenticate_user");
    assert_eq!(results[0].match_type, MatchType::Hybrid);
    assert!(
        results
            .iter()
            .any(|result| result.match_type == MatchType::Vector)
    );
    assert!(results[0].combined_score > results[1].combined_score);
}

#[tokio::test]
async fn search_hybrid_uses_vector_results_when_query_has_no_fts_tokens() {
    let (_temp_dir, database) = open_temp_database().await;
    let target = IndexedCodeUnit {
        unit: sample_unit(
            "semantic_auth",
            "src/semantic.rs",
            10,
            "pub fn semantic_auth() -> bool { true }",
        ),
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        file_hash: None,
    };

    database
        .store_indexed_units(REPO_ID, BRANCH, &[target])
        .await
        .unwrap();

    let results = database
        .search_hybrid(REPO_ID, BRANCH, "   ", &[1.0, 0.0, 0.0, 0.0], 1, 60)
        .await
        .unwrap();

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].unit.name, "semantic_auth");
    assert_eq!(results[0].match_type, MatchType::Vector);
}

#[tokio::test]
async fn search_hybrid_fallback_tracks_source_specific_rrf_scores() {
    let (_temp_dir, database) = open_temp_database().await;
    let hybrid = IndexedCodeUnit {
        unit: sample_unit(
            "authenticate_user",
            "src/auth.rs",
            10,
            "pub fn authenticate_user() -> bool { true }",
        ),
        embedding: Some(vec![1.0, 0.0, 0.0, 0.0]),
        file_hash: None,
    };
    let vector_only = IndexedCodeUnit {
        unit: sample_unit(
            "semantic_auth",
            "src/semantic.rs",
            30,
            "pub fn semantic_auth() -> bool { false }",
        ),
        embedding: Some(vec![0.95, 0.05, 0.0, 0.0]),
        file_hash: None,
    };

    database
        .store_indexed_units(REPO_ID, BRANCH, &[hybrid, vector_only])
        .await
        .unwrap();

    let results = database
        .search_hybrid_fallback(
            REPO_ID,
            BRANCH,
            "authenticate user",
            &[1.0, 0.0, 0.0, 0.0],
            2,
            60,
        )
        .await
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].unit.name, "authenticate_user");
    assert_eq!(results[0].match_type, MatchType::Hybrid);
    assert!(results[0].fts_score.unwrap() > 0.0);
    assert!(results[0].vector_score.unwrap() > 0.0);
    assert!(results[0].combined_score > results[0].fts_score.unwrap());
    assert!(results[0].combined_score > results[0].vector_score.unwrap());
}
