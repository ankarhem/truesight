use super::*;

pub(super) fn normalize_repo_root(repo_root: &Path) -> std::result::Result<PathBuf, DatabaseError> {
    if repo_root.exists() {
        return Ok(repo_root.canonicalize()?);
    }

    if repo_root.is_absolute() {
        Ok(repo_root.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(repo_root))
    }
}

pub(super) fn sanitize_path_segment(value: &str) -> String {
    let mut sanitized = String::with_capacity(value.len());

    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            sanitized.push(ch);
        } else {
            sanitized.push('-');
        }
    }

    let trimmed = sanitized.trim_matches('-');
    if trimmed.is_empty() {
        String::from("repo")
    } else {
        trimmed.to_owned()
    }
}

pub(super) fn short_hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();

    let mut hex = String::with_capacity(16);
    for byte in digest.iter().take(8) {
        let _ = write!(&mut hex, "{byte:02x}");
    }

    hex
}

pub(super) fn long_hex_digest(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();

    let mut hex = String::with_capacity(64);
    for byte in digest {
        let _ = write!(&mut hex, "{byte:02x}");
    }

    hex
}

pub(super) fn code_unit_id(repo_id: &str, branch: &str, unit: &CodeUnit) -> String {
    long_hex_digest(
        format!(
            "{repo_id}\u{1f}{branch}\u{1f}{}\u{1f}{}\u{1f}{}\u{1f}{}",
            path_to_string(&unit.file_path),
            unit.name,
            unit.kind,
            unit.line_start,
        )
        .as_bytes(),
    )
}

pub(super) fn default_file_hash(unit: &CodeUnit) -> String {
    long_hex_digest(unit.content.as_bytes())
}

pub(super) fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

pub(super) fn encode_embedding(embedding: &[f32]) -> std::result::Result<Vec<u8>, DatabaseError> {
    if embedding.is_empty() {
        return Err(DatabaseError::InvalidEmbedding(
            "embedding must not be empty".to_string(),
        ));
    }

    let mut blob = Vec::with_capacity(embedding.len() * 4);
    for value in embedding {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    Ok(blob)
}

pub(super) fn decode_embedding(blob: &[u8]) -> Result<Vec<f32>> {
    if !blob.len().is_multiple_of(4) {
        return Err(DatabaseError::InvalidEmbeddingLength(blob.len()).into());
    }

    let mut embedding = Vec::with_capacity(blob.len() / 4);
    for chunk in blob.chunks_exact(4) {
        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
        embedding.push(f32::from_le_bytes(bytes));
    }
    Ok(embedding)
}

pub(super) fn cosine_similarity(left: &[f32], right: &[f32]) -> Result<f32> {
    if left.len() != right.len() {
        return Err(DatabaseError::InvalidEmbedding(format!(
            "embedding dimension mismatch: {} != {}",
            left.len(),
            right.len()
        ))
        .into());
    }

    let mut dot = 0.0f32;
    let mut left_norm = 0.0f32;
    let mut right_norm = 0.0f32;

    for (left_value, right_value) in left.iter().zip(right.iter()) {
        dot += left_value * right_value;
        left_norm += left_value * left_value;
        right_norm += right_value * right_value;
    }

    if left_norm == 0.0 || right_norm == 0.0 {
        return Ok(0.0);
    }

    Ok(dot / (left_norm.sqrt() * right_norm.sqrt()))
}

pub(super) fn sanitize_fts_query(query: &str) -> Option<String> {
    let tokens: Vec<String> = query
        .split_whitespace()
        .map(|word| {
            let escaped = word.replace('"', "\"\"");
            format!("\"{}\"", escaped)
        })
        .collect();

    if tokens.is_empty() {
        None
    } else {
        Some(tokens.join(" "))
    }
}

pub(super) fn normalize_fts_score(raw_score: f32) -> f32 {
    let magnitude = if raw_score.is_sign_negative() {
        -raw_score
    } else {
        raw_score
    };
    1.0 / (1.0 + magnitude)
}

pub(super) fn hybrid_candidate_limit(limit: usize) -> usize {
    limit.max(1).saturating_mul(3)
}

pub(super) fn parse_timestamp(value: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .map(|datetime| datetime.with_timezone(&Utc))
        .map_err(|_| DatabaseError::InvalidTimestamp(value.to_string()).into())
}
