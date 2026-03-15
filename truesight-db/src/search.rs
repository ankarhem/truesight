use super::*;

#[derive(Clone)]
struct FusedResult {
    unit: CodeUnit,
    rrf_score: f32,
    fts_score: f32,
    vector_score: f32,
    saw_fts: bool,
    saw_vector: bool,
    match_type: MatchType,
}

pub(super) fn ranked_result_from_row(
    row: &libsql::Row,
    fts_score_offset: i32,
    vector_score_offset: i32,
    combined_score_offset: i32,
    match_type_offset: i32,
) -> std::result::Result<RankedResult, DatabaseError> {
    let match_type = row
        .get::<String>(match_type_offset)
        .map_err(DatabaseError::from)?
        .parse()
        .map_err(DatabaseError::InvalidEnumValue)?;

    Ok(RankedResult {
        unit: code_unit_from_row(row, 0)
            .map_err(|error| DatabaseError::InvalidEmbedding(error.to_string()))?,
        fts_score: row
            .get::<Option<f64>>(fts_score_offset)
            .map_err(DatabaseError::from)?
            .map(|score| score as f32),
        vector_score: row
            .get::<Option<f64>>(vector_score_offset)
            .map_err(DatabaseError::from)?
            .map(|score| score as f32),
        combined_score: row
            .get::<f64>(combined_score_offset)
            .map_err(DatabaseError::from)? as f32,
        match_type,
    })
}

pub(super) fn fuse_ranked_results(
    fts_results: Vec<RankedResult>,
    vector_results: Vec<RankedResult>,
    rrf_k: u32,
    limit: usize,
) -> Vec<RankedResult> {
    let mut fused: HashMap<String, FusedResult> = HashMap::new();
    let rrf_k = rrf_k.max(1) as f32;

    accumulate_rrf(&mut fused, fts_results, rrf_k, MatchType::Fts);
    accumulate_rrf(&mut fused, vector_results, rrf_k, MatchType::Vector);

    let mut results = fused
        .into_values()
        .map(|result| RankedResult {
            unit: result.unit,
            fts_score: if result.saw_fts {
                Some(result.fts_score)
            } else {
                None
            },
            vector_score: if result.saw_vector {
                Some(result.vector_score)
            } else {
                None
            },
            combined_score: result.rrf_score,
            match_type: result.match_type,
        })
        .collect::<Vec<_>>();

    results.sort_by(compare_ranked_results);
    results.truncate(limit);
    results
}

fn accumulate_rrf(
    fused: &mut HashMap<String, FusedResult>,
    results: Vec<RankedResult>,
    rrf_k: f32,
    source: MatchType,
) {
    for (index, result) in results.into_iter().enumerate() {
        let key = ranked_result_key(&result.unit);
        let rank = (index + 1) as f32;
        let contribution = 1.0 / (rrf_k + rank);

        let entry = fused.entry(key).or_insert_with(|| FusedResult {
            unit: result.unit.clone(),
            rrf_score: 0.0,
            fts_score: 0.0,
            vector_score: 0.0,
            saw_fts: false,
            saw_vector: false,
            match_type: source,
        });

        entry.rrf_score += contribution;

        match source {
            MatchType::Fts => {
                entry.saw_fts = true;
                entry.fts_score += contribution;
            }
            MatchType::Vector => {
                entry.saw_vector = true;
                entry.vector_score += contribution;
            }
            MatchType::Hybrid => {}
        }

        entry.match_type = match (entry.saw_fts, entry.saw_vector) {
            (true, true) => MatchType::Hybrid,
            (true, false) => MatchType::Fts,
            (false, true) => MatchType::Vector,
            (false, false) => source,
        };
    }
}

fn compare_ranked_results(left: &RankedResult, right: &RankedResult) -> Ordering {
    right
        .combined_score
        .partial_cmp(&left.combined_score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| match_type_rank(right.match_type).cmp(&match_type_rank(left.match_type)))
        .then_with(|| left.unit.line_start.cmp(&right.unit.line_start))
        .then_with(|| left.unit.file_path.cmp(&right.unit.file_path))
        .then_with(|| left.unit.name.cmp(&right.unit.name))
}

fn match_type_rank(match_type: MatchType) -> u8 {
    match match_type {
        MatchType::Hybrid => 2,
        MatchType::Fts | MatchType::Vector => 1,
    }
}

fn ranked_result_key(unit: &CodeUnit) -> String {
    format!(
        "{}\u{1f}{}\u{1f}{:?}\u{1f}{}",
        unit.file_path.display(),
        unit.name,
        unit.kind,
        unit.line_start,
    )
}
