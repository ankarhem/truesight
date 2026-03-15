use super::*;

pub(super) const UPSERT_CODE_UNIT_SQL: &str = r#"
    INSERT INTO code_units (
        id, repo_id, branch, file_path, name, kind, signature,
        doc, content, parent, language, line_start, line_end,
        embedding, embedding_model, file_hash
    )
    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)
    ON CONFLICT(repo_id, branch, file_path, name, kind, line_start) DO UPDATE SET
        id = excluded.id,
        signature = excluded.signature,
        doc = excluded.doc,
        content = excluded.content,
        parent = excluded.parent,
        language = excluded.language,
        line_end = excluded.line_end,
        embedding = CASE
            WHEN code_units.signature != excluded.signature
              OR ifnull(code_units.doc, '') != ifnull(excluded.doc, '')
              OR code_units.content != excluded.content
            THEN NULL
            WHEN excluded.embedding IS NOT NULL THEN excluded.embedding
            ELSE code_units.embedding
        END,
        embedding_model = CASE
            WHEN code_units.signature != excluded.signature
              OR ifnull(code_units.doc, '') != ifnull(excluded.doc, '')
              OR code_units.content != excluded.content
            THEN ''
            WHEN excluded.embedding_model != '' THEN excluded.embedding_model
            ELSE code_units.embedding_model
        END,
        file_hash = excluded.file_hash
"#;

pub(super) async fn insert_code_unit(
    tx: &libsql::Transaction,
    repo_id: &str,
    branch: &str,
    entry: &IndexedCodeUnit,
) -> std::result::Result<(), DatabaseError> {
    let unit_id = code_unit_id(repo_id, branch, &entry.unit);
    let embedding_blob = entry
        .embedding
        .as_deref()
        .map(encode_embedding)
        .transpose()?;
    let file_hash = entry
        .file_hash
        .clone()
        .unwrap_or_else(|| default_file_hash(&entry.unit));

    tx.execute(
        UPSERT_CODE_UNIT_SQL,
        params![
            unit_id,
            repo_id,
            branch,
            path_to_string(&entry.unit.file_path),
            entry.unit.name.clone(),
            entry.unit.kind.to_string(),
            entry.unit.signature.clone(),
            entry.unit.doc.clone(),
            entry.unit.content.clone(),
            entry.unit.parent.clone(),
            entry.unit.language.to_string(),
            i64::from(entry.unit.line_start),
            i64::from(entry.unit.line_end),
            embedding_blob,
            if entry.embedding.is_some() {
                "unknown"
            } else {
                ""
            },
            file_hash,
        ],
    )
    .await?;
    Ok(())
}

pub(super) fn code_unit_from_row(row: &libsql::Row, offset: i32) -> Result<CodeUnit> {
    let kind_str = row.get::<String>(offset + 1).map_err(DatabaseError::from)?;
    let lang_str = row.get::<String>(offset + 9).map_err(DatabaseError::from)?;

    Ok(CodeUnit {
        name: row.get::<String>(offset).map_err(DatabaseError::from)?,
        kind: kind_str
            .parse()
            .map_err(|err: String| DatabaseError::InvalidEnumValue(err))?,
        signature: row.get::<String>(offset + 2).map_err(DatabaseError::from)?,
        doc: row
            .get::<Option<String>>(offset + 3)
            .map_err(DatabaseError::from)?,
        file_path: PathBuf::from(row.get::<String>(offset + 4).map_err(DatabaseError::from)?),
        line_start: row.get::<i64>(offset + 5).map_err(DatabaseError::from)? as u32,
        line_end: row.get::<i64>(offset + 6).map_err(DatabaseError::from)? as u32,
        content: row.get::<String>(offset + 7).map_err(DatabaseError::from)?,
        parent: row
            .get::<Option<String>>(offset + 8)
            .map_err(DatabaseError::from)?,
        language: lang_str
            .parse()
            .map_err(|err: String| DatabaseError::InvalidEnumValue(err))?,
    })
}
