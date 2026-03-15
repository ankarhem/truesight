use super::*;

#[tokio::test]
async fn delete_units_for_file_removes_only_requested_file() {
    let (_temp_dir, database) = open_temp_database().await;
    let alpha = sample_unit(
        "alpha_handler",
        "src/alpha.rs",
        10,
        "pub fn alpha_handler() -> bool { true }",
    );
    let alpha_helper = sample_unit(
        "alpha_helper",
        "src/alpha.rs",
        14,
        "pub fn alpha_helper() -> bool { true }",
    );
    let beta = sample_unit(
        "beta_handler",
        "src/beta.rs",
        20,
        "pub fn beta_handler() -> bool { false }",
    );

    database
        .store_code_units(REPO_ID, BRANCH, &[alpha, alpha_helper, beta])
        .await
        .unwrap();

    database
        .delete_units_for_file(REPO_ID, BRANCH, Path::new("src/alpha.rs"))
        .await
        .unwrap();

    let stored = database.get_all_symbols(REPO_ID, BRANCH).await.unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0].name, "beta_handler");
}

#[tokio::test]
async fn index_metadata_round_trips() {
    let (_temp_dir, database) = open_temp_database().await;
    let metadata = sample_metadata(BRANCH, Some("abc123"), 7, 21);

    database
        .set_index_metadata(REPO_ID, BRANCH, &metadata)
        .await
        .unwrap();

    let stored = database
        .get_index_metadata(REPO_ID, BRANCH)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(stored.repo_id, metadata.repo_id);
    assert_eq!(stored.branch, metadata.branch);
    assert_eq!(stored.last_commit_sha, metadata.last_commit_sha);
    assert_eq!(stored.file_count, metadata.file_count);
    assert_eq!(stored.symbol_count, metadata.symbol_count);

    let missing = database
        .get_index_metadata(REPO_ID, "feature")
        .await
        .unwrap();
    assert!(missing.is_none());
}

#[tokio::test]
async fn indexed_files_round_trip_and_branch_delete_cleans_related_rows() {
    let (_temp_dir, database) = open_temp_database().await;
    database
        .upsert_indexed_file(REPO_ID, BRANCH, Path::new("src/alpha.rs"), "hash-alpha", 1)
        .await
        .unwrap();
    database
        .upsert_indexed_file(REPO_ID, BRANCH, Path::new("src/beta.rs"), "hash-beta", 2)
        .await
        .unwrap();
    database
        .set_index_metadata(REPO_ID, BRANCH, &sample_metadata(BRANCH, None, 2, 3))
        .await
        .unwrap();
    database
        .store_code_units(
            REPO_ID,
            BRANCH,
            &[sample_unit(
                "alpha_handler",
                "src/alpha.rs",
                10,
                "pub fn alpha_handler() -> bool { true }",
            )],
        )
        .await
        .unwrap();

    let files = database.get_indexed_files(REPO_ID, BRANCH).await.unwrap();
    assert_eq!(files.len(), 2);
    assert_eq!(files[0].file_path, PathBuf::from("src/alpha.rs"));
    assert_eq!(files[0].file_hash, "hash-alpha");
    assert!(matches!(files[0], IndexedFileRecord { .. }));

    database.delete_branch_index(REPO_ID, BRANCH).await.unwrap();

    assert!(
        database
            .get_all_symbols(REPO_ID, BRANCH)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        database
            .get_indexed_files(REPO_ID, BRANCH)
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        database
            .get_index_metadata(REPO_ID, BRANCH)
            .await
            .unwrap()
            .is_none()
    );
}

#[tokio::test]
async fn list_branches_returns_distinct_sorted_branch_names() {
    let (_temp_dir, database) = open_temp_database().await;

    for branch in ["feature/auth", "main", "feature/auth", "detached-deadbeef"] {
        database
            .set_index_metadata(REPO_ID, branch, &sample_metadata(branch, None, 0, 0))
            .await
            .unwrap();
    }

    let branches = database.list_branches(REPO_ID).await.unwrap();

    assert_eq!(
        branches,
        vec![
            "detached-deadbeef".to_string(),
            "feature/auth".to_string(),
            "main".to_string(),
        ]
    );
}

#[tokio::test]
async fn branch_scoped_queries_keep_metadata_and_symbols_isolated() {
    let (_temp_dir, database) = open_temp_database().await;
    let main_unit = sample_unit(
        "main_only",
        "src/lib.rs",
        10,
        "pub fn main_only() -> bool { true }",
    );
    let feature_unit = sample_unit(
        "feature_only",
        "src/lib.rs",
        10,
        "pub fn feature_only() -> bool { true }",
    );

    database
        .store_code_units(REPO_ID, "main", std::slice::from_ref(&main_unit))
        .await
        .unwrap();
    database
        .store_code_units(REPO_ID, "feature/auth", std::slice::from_ref(&feature_unit))
        .await
        .unwrap();
    database
        .set_index_metadata(
            REPO_ID,
            "main",
            &sample_metadata("main", Some("1111111"), 1, 1),
        )
        .await
        .unwrap();
    database
        .set_index_metadata(
            REPO_ID,
            "feature/auth",
            &sample_metadata("feature/auth", Some("2222222"), 1, 1),
        )
        .await
        .unwrap();

    let main_symbols = database.get_all_symbols(REPO_ID, "main").await.unwrap();
    let feature_symbols = database
        .get_all_symbols(REPO_ID, "feature/auth")
        .await
        .unwrap();
    let main_search = database
        .search_fts(REPO_ID, "main", "main_only", 5)
        .await
        .unwrap();
    let feature_search = database
        .search_fts(REPO_ID, "feature/auth", "feature_only", 5)
        .await
        .unwrap();
    let missing_cross_branch = database
        .search_fts(REPO_ID, "main", "feature_only", 5)
        .await
        .unwrap();

    assert_eq!(main_symbols.len(), 1);
    assert_eq!(feature_symbols.len(), 1);
    assert_eq!(main_symbols[0].name, main_unit.name);
    assert_eq!(feature_symbols[0].name, feature_unit.name);
    assert_eq!(main_search.len(), 1);
    assert_eq!(main_search[0].unit.name, "main_only");
    assert_eq!(feature_search.len(), 1);
    assert_eq!(feature_search[0].unit.name, "feature_only");
    assert!(missing_cross_branch.is_empty());
    assert_eq!(
        database
            .get_index_metadata(REPO_ID, "main")
            .await
            .unwrap()
            .unwrap()
            .last_commit_sha,
        Some("1111111".to_string())
    );
    assert_eq!(
        database
            .get_index_metadata(REPO_ID, "feature/auth")
            .await
            .unwrap()
            .unwrap()
            .last_commit_sha,
        Some("2222222".to_string())
    );
}

#[tokio::test]
async fn cleanup_branch_removes_only_requested_partition() {
    let (_temp_dir, database) = open_temp_database().await;

    database
        .store_code_units(
            REPO_ID,
            "main",
            &[sample_unit(
                "main_only",
                "src/lib.rs",
                10,
                "pub fn main_only() -> bool { true }",
            )],
        )
        .await
        .unwrap();
    database
        .store_code_units(
            REPO_ID,
            "feature/auth",
            &[sample_unit(
                "feature_only",
                "src/lib.rs",
                10,
                "pub fn feature_only() -> bool { true }",
            )],
        )
        .await
        .unwrap();
    database
        .upsert_indexed_file(REPO_ID, "main", Path::new("src/lib.rs"), "hash-main", 1)
        .await
        .unwrap();
    database
        .upsert_indexed_file(
            REPO_ID,
            "feature/auth",
            Path::new("src/lib.rs"),
            "hash-feature",
            1,
        )
        .await
        .unwrap();
    database
        .set_index_metadata(REPO_ID, "main", &sample_metadata("main", None, 1, 1))
        .await
        .unwrap();
    database
        .set_index_metadata(
            REPO_ID,
            "feature/auth",
            &sample_metadata("feature/auth", None, 1, 1),
        )
        .await
        .unwrap();

    database
        .cleanup_branch(REPO_ID, "feature/auth")
        .await
        .unwrap();

    assert_eq!(
        database
            .get_all_symbols(REPO_ID, "main")
            .await
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        database
            .get_indexed_files(REPO_ID, "main")
            .await
            .unwrap()
            .len(),
        1
    );
    assert!(
        database
            .get_index_metadata(REPO_ID, "main")
            .await
            .unwrap()
            .is_some()
    );
    assert!(
        database
            .get_all_symbols(REPO_ID, "feature/auth")
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        database
            .get_indexed_files(REPO_ID, "feature/auth")
            .await
            .unwrap()
            .is_empty()
    );
    assert!(
        database
            .get_index_metadata(REPO_ID, "feature/auth")
            .await
            .unwrap()
            .is_none()
    );
}

#[test]
fn embedding_helpers_round_trip_and_reject_invalid_inputs() {
    let blob = encode_embedding(&[1.0, 2.5, -3.0, 4.25]).unwrap();
    assert_eq!(decode_embedding(&blob).unwrap(), vec![1.0, 2.5, -3.0, 4.25]);

    let invalid = decode_embedding(&[1, 2, 3]).unwrap_err();
    assert!(
        invalid
            .to_string()
            .contains("invalid embedding blob length")
    );
    assert!(encode_embedding(&[]).is_err());
}

#[test]
fn cosine_similarity_handles_exact_and_mismatched_vectors() {
    let exact = cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]).unwrap();
    assert!((exact - 1.0).abs() < f32::EPSILON);

    let orthogonal = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
    assert!(orthogonal.abs() < f32::EPSILON);

    let mismatched = cosine_similarity(&[1.0], &[1.0, 0.0]).unwrap_err();
    assert!(
        mismatched
            .to_string()
            .contains("embedding dimension mismatch")
    );
}
