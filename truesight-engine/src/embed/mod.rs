use ndarray::Array2;
use ort::{
    ep,
    session::{Session, builder::GraphOptimizationLevel},
};
use std::sync::{Arc, Mutex, OnceLock};
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};
use truesight_core::{Embedder, Result, TruesightError};

mod assets;
mod encoding;
mod inference;

const MODEL_NAME: &str = "all-MiniLM-L6-v2";
const MODEL_DIMENSION: usize = 384;
const MAX_SEQUENCE_LENGTH: usize = 512;
const MODEL_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json";
const MODEL_SHA256: &str = "6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452";
const TOKENIZER_SHA256: &str = "be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037";
const SENTENCE_OUTPUT_NAMES: &[&str] = &["sentence_embedding", "sentence_embeddings"];
const TOKEN_OUTPUT_NAMES: &[&str] = &["token_embeddings", "last_hidden_state", "embeddings"];

pub struct OnnxEmbedder {
    inner: Arc<OnnxEmbedderInner>,
}

struct OnnxEmbedderInner {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    includes_token_type_ids: bool,
}

static EMBEDDER_CACHE: OnceLock<Mutex<Option<Arc<OnnxEmbedderInner>>>> = OnceLock::new();

impl OnnxEmbedder {
    pub fn new() -> Result<Self> {
        let cache = EMBEDDER_CACHE.get_or_init(|| Mutex::new(None));
        let mut cached = cache
            .lock()
            .map_err(|_| TruesightError::Embedding("failed to lock embedder cache".to_string()))?;

        if let Some(inner) = cached.as_ref() {
            return Ok(Self {
                inner: inner.clone(),
            });
        }

        let assets = Self::ensure_model()?;
        let inner = Arc::new(Self::build_inner(&assets)?);
        *cached = Some(inner.clone());

        Ok(Self { inner })
    }

    fn build_inner(assets: &assets::ModelAssets) -> Result<OnnxEmbedderInner> {
        let mut tokenizer = Tokenizer::from_file(&assets.tokenizer).map_err(|error| {
            TruesightError::Embedding(format!("failed to load tokenizer: {error}"))
        })?;

        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQUENCE_LENGTH,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                direction: TruncationDirection::Right,
            }))
            .map_err(|error| {
                TruesightError::Embedding(format!(
                    "failed to configure tokenizer truncation: {error}"
                ))
            })?;

        let pad_id = tokenizer.token_to_id("[PAD]").unwrap_or(0);
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            direction: PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token: "[PAD]".to_string(),
        }));

        let session = Session::builder()
            .map_err(|error| {
                TruesightError::Embedding(format!("failed to create ONNX session builder: {error}"))
            })?
            .with_parallel_execution(false)
            .map_err(|error| {
                TruesightError::Embedding(format!(
                    "failed to configure ONNX execution mode: {error}"
                ))
            })?
            .with_optimization_level(GraphOptimizationLevel::Disable)
            .map_err(|error| {
                TruesightError::Embedding(format!(
                    "failed to configure ONNX graph optimization: {error}"
                ))
            })?
            .with_no_environment_execution_providers()
            .map_err(|error| {
                TruesightError::Embedding(format!(
                    "failed to isolate ONNX environment providers: {error}"
                ))
            })?
            .with_execution_providers([ep::CPU::default().build()])
            .map_err(|error| {
                TruesightError::Embedding(format!(
                    "failed to configure ONNX execution providers: {error}"
                ))
            })?
            .with_intra_threads(1)
            .map_err(|error| {
                TruesightError::Embedding(format!(
                    "failed to configure ONNX intra-op threads: {error}"
                ))
            })?
            .with_inter_threads(1)
            .map_err(|error| {
                TruesightError::Embedding(format!(
                    "failed to configure ONNX inter-op threads: {error}"
                ))
            })?
            .commit_from_file(&assets.model)
            .map_err(|error| {
                TruesightError::Embedding(format!("failed to load ONNX model: {error}"))
            })?;

        let includes_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        Ok(OnnxEmbedderInner {
            session: Mutex::new(session),
            tokenizer,
            includes_token_type_ids,
        })
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        batch.into_iter().next().ok_or_else(|| {
            TruesightError::Embedding("embedder returned an empty batch".to_string())
        })
    }

    fn embed_batch<'a>(&self, texts: &[&'a str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encoded = self.encode_batch(texts)?;
        let embeddings = self.infer(&encoded)?;

        if embeddings
            .iter()
            .any(|embedding| embedding.len() != MODEL_DIMENSION)
        {
            return Err(TruesightError::Embedding(format!(
                "unexpected embedding dimension, expected {MODEL_DIMENSION}"
            )));
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        MODEL_DIMENSION
    }

    fn model_name(&self) -> &str {
        MODEL_NAME
    }
}

struct EncodedBatch {
    input_ids: Array2<i64>,
    attention_mask: Array2<i64>,
    token_type_ids: Array2<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn embedder() -> Arc<OnnxEmbedder> {
        Arc::new(OnnxEmbedder::new().expect("embedder should initialize"))
    }

    fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
        left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn test_embed_basic() {
        let embedder = embedder();
        let embedding = embedder
            .embed("hello world")
            .expect("embedding should succeed");

        assert_eq!(embedding.len(), MODEL_DIMENSION);
        assert!(
            OnnxEmbedder::model_dir()
                .unwrap()
                .join("model.onnx")
                .exists()
        );
        assert!(
            OnnxEmbedder::model_dir()
                .unwrap()
                .join("tokenizer.json")
                .exists()
        );
    }

    #[test]
    fn test_embed_is_deterministic() {
        let embedder = embedder();
        let first = embedder
            .embed("hello world")
            .expect("first embedding should succeed");
        let second = embedder
            .embed("hello world")
            .expect("second embedding should succeed");

        assert_eq!(first, second);
    }

    #[test]
    fn test_embed_batch() {
        let embedder = embedder();
        let inputs = ["rust function", "rust method", "cooking recipe"];
        let batch = embedder
            .embed_batch(&inputs)
            .expect("batch embedding should succeed");

        assert_eq!(batch.len(), inputs.len());
        assert!(
            batch
                .iter()
                .all(|embedding| embedding.len() == MODEL_DIMENSION)
        );
        assert_eq!(
            batch[0],
            embedder
                .embed(inputs[0])
                .expect("single embedding should succeed")
        );
    }

    #[test]
    fn test_embed_quality() {
        let embedder = embedder();
        let function = embedder
            .embed("rust function")
            .expect("function embedding should succeed");
        let method = embedder
            .embed("rust method")
            .expect("method embedding should succeed");
        let recipe = embedder
            .embed("cooking recipe")
            .expect("recipe embedding should succeed");

        let related_similarity = cosine_similarity(&function, &method);
        let unrelated_similarity = cosine_similarity(&function, &recipe);

        assert!(related_similarity > unrelated_similarity);
    }

    #[test]
    fn test_dimension() {
        assert_eq!(embedder().dimension(), MODEL_DIMENSION);
    }

    #[test]
    fn test_new_reuses_cached_runtime() {
        let first = OnnxEmbedder::new().expect("first embedder should initialize");
        let second = OnnxEmbedder::new().expect("second embedder should reuse cache");

        assert!(Arc::ptr_eq(&first.inner, &second.inner));
    }
}
