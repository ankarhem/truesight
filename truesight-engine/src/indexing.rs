use std::path::{Path, PathBuf};

use truesight_core::{
    CodeUnit, Embedder, EmbeddingUpdate, IndexStorage, IndexedCodeUnit, Result, TruesightError,
};

const EMBEDDING_BATCH_SIZE: usize = 128;

#[cfg(test)]
pub(crate) fn build_indexed_units<E: Embedder + ?Sized>(
    units: Vec<CodeUnit>,
    file_hash: &str,
    embedder: &E,
) -> Result<Vec<IndexedCodeUnit>> {
    units
        .into_iter()
        .map(|unit| {
            let embedding = embedder.embed(&embedding_text(&unit))?;
            Ok(IndexedCodeUnit {
                unit,
                embedding: Some(embedding),
                file_hash: Some(file_hash.to_string()),
            })
        })
        .collect()
}

pub(crate) fn build_pending_units(units: Vec<CodeUnit>, file_hash: &str) -> Vec<IndexedCodeUnit> {
    units
        .into_iter()
        .map(|unit| IndexedCodeUnit {
            unit,
            embedding: None,
            file_hash: Some(file_hash.to_string()),
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn embedding_text(unit: &CodeUnit) -> String {
    match unit.doc.as_deref() {
        Some(doc) if !doc.is_empty() => format!("{} {} {}", unit.signature, doc, unit.content),
        _ => format!("{} {}", unit.signature, unit.content),
    }
}

pub(crate) async fn materialize_embeddings<S, E>(
    storage: &S,
    repo_id: &str,
    branch: &str,
    embedder: &E,
) -> Result<u32>
where
    S: IndexStorage + ?Sized,
    E: Embedder + ?Sized,
{
    let mut chunks_embedded = 0_u32;

    loop {
        let pending = storage
            .list_pending_embeddings(repo_id, branch, embedder.model_name(), EMBEDDING_BATCH_SIZE)
            .await?;

        if pending.is_empty() {
            return Ok(chunks_embedded);
        }

        let texts = pending
            .iter()
            .map(|entry| entry.embedding_text())
            .collect::<Vec<_>>();
        let text_refs = texts.iter().map(String::as_str).collect::<Vec<_>>();
        let embeddings = embedder.embed_batch(&text_refs)?;

        let updates = pending
            .into_iter()
            .zip(embeddings)
            .map(|(entry, embedding)| EmbeddingUpdate {
                id: entry.id,
                embedding,
            })
            .collect::<Vec<_>>();

        chunks_embedded += updates.len() as u32;
        storage
            .update_embeddings(repo_id, branch, embedder.model_name(), &updates)
            .await?;
    }
}

pub(crate) fn repo_relative_path(root: &Path, path: &Path) -> Result<PathBuf> {
    path.strip_prefix(root)
        .map(|relative| relative.to_path_buf())
        .map_err(|_| {
            TruesightError::Index(format!(
                "{} is outside repository root {}",
                path.display(),
                root.display()
            ))
        })
}

#[cfg(test)]
mod tests {
    use std::path::{Path, PathBuf};

    use truesight_core::{CodeUnitKind, Language};

    use super::*;

    struct RecordingEmbedder;

    impl Embedder for RecordingEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(vec![text.len() as f32])
        }

        fn embed_batch<'a>(&self, texts: &[&'a str]) -> Result<Vec<Vec<f32>>> {
            texts.iter().map(|text| self.embed(text)).collect()
        }

        fn dimension(&self) -> usize {
            1
        }
    }

    #[test]
    fn build_indexed_units_uses_canonical_embedding_text() {
        let unit = CodeUnit {
            name: "alpha".to_string(),
            kind: CodeUnitKind::Function,
            signature: "pub fn alpha(value: usize) -> bool".to_string(),
            doc: Some("Checks whether the value is valid.".to_string()),
            file_path: PathBuf::from("src/lib.rs"),
            line_start: 1,
            line_end: 3,
            content: "pub fn alpha(value: usize) -> bool { value > 0 }".to_string(),
            parent: None,
            language: Language::Rust,
        };

        let indexed = build_indexed_units(vec![unit.clone()], "hash-123", &RecordingEmbedder)
            .expect("indexed unit should be created");

        assert_eq!(indexed.len(), 1);
        assert_eq!(indexed[0].file_hash.as_deref(), Some("hash-123"));
        assert_eq!(indexed[0].unit.file_path, PathBuf::from("src/lib.rs"));
        assert_eq!(
            indexed[0].embedding,
            Some(vec![embedding_text(&unit).len() as f32])
        );
    }

    #[test]
    fn repo_relative_path_rejects_paths_outside_root() {
        let error = repo_relative_path(Path::new("/repo"), Path::new("/elsewhere/src/lib.rs"))
            .expect_err("path outside root should fail");

        assert!(matches!(error, TruesightError::Index(_)));
    }
}
