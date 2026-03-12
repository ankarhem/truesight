use ndarray::{Array2, ArrayView2, ArrayView3, Axis, Ix2, Ix3};
use ort::{
    ep,
    session::{Session, SessionOutputs, builder::GraphOptimizationLevel},
    value::TensorRef,
};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use tokenizers::{
    PaddingDirection, PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection,
    TruncationParams, TruncationStrategy,
};
use truesight_core::{Embedder, Result, TruesightError};

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
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    includes_token_type_ids: bool,
}

impl OnnxEmbedder {
    pub fn new() -> Result<Self> {
        let assets = Self::ensure_model()?;
        Self::initialize_ort_environment()?;
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

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            includes_token_type_ids,
        })
    }

    pub fn model_dir() -> Result<PathBuf> {
        Ok(Self::config_dir()?.join("models").join(MODEL_NAME))
    }

    fn config_dir() -> Result<PathBuf> {
        if let Some(override_dir) = env::var_os("TRUESIGHT_CONFIG_DIR") {
            return Ok(PathBuf::from(override_dir));
        }

        let home = env::var_os("HOME").ok_or_else(|| {
            TruesightError::Embedding("HOME environment variable is not set".to_string())
        })?;

        Ok(PathBuf::from(home).join(".config").join("truesight"))
    }

    fn ensure_model() -> Result<ModelAssets> {
        let model_dir = Self::model_dir()?;
        fs::create_dir_all(&model_dir)?;

        let model = model_dir.join("model.onnx");
        let tokenizer = model_dir.join("tokenizer.json");

        Self::ensure_file(&model, MODEL_URL, MODEL_SHA256)?;
        Self::ensure_file(&tokenizer, TOKENIZER_URL, TOKENIZER_SHA256)?;

        Ok(ModelAssets { model, tokenizer })
    }

    fn initialize_ort_environment() -> Result<()> {
        let (path, source) = Self::resolve_ort_dylib_path()?;
        env::set_var("ORT_DYLIB_PATH", &path);
        let _ = source;
        Ok(())
    }

    fn resolve_ort_dylib_path() -> Result<(PathBuf, &'static str)> {
        if let Some(path) = env::var_os("ORT_DYLIB_PATH") {
            return Ok((
                Self::validate_ort_dylib_path(PathBuf::from(path), "ORT_DYLIB_PATH")?,
                "ORT_DYLIB_PATH",
            ));
        }

        if let Some(path) = option_env!("TRUESIGHT_ORT_DYLIB") {
            return Ok((
                Self::validate_ort_dylib_path(PathBuf::from(path), "TRUESIGHT_ORT_DYLIB")?,
                "TRUESIGHT_ORT_DYLIB",
            ));
        }

        Err(TruesightError::Embedding(
            "ONNX Runtime dylib path is not configured; set ORT_DYLIB_PATH or rebuild with a valid TRUESIGHT_ORT_DYLIB"
                .to_string(),
        ))
    }

    fn validate_ort_dylib_path(path: PathBuf, source: &str) -> Result<PathBuf> {
        let canonical = path.canonicalize().map_err(|error| {
            TruesightError::Embedding(format!(
                "failed to resolve ONNX Runtime dylib from {source} at {}: {error}",
                path.display()
            ))
        })?;

        if canonical.is_file() {
            Ok(canonical)
        } else {
            Err(TruesightError::Embedding(format!(
                "resolved ONNX Runtime dylib from {source} is not a file: {}",
                canonical.display()
            )))
        }
    }

    fn ensure_file(path: &Path, url: &str, expected_sha256: &str) -> Result<()> {
        if path.exists() {
            let actual_sha256 = Self::sha256_file(path)?;
            if actual_sha256 == expected_sha256 {
                return Ok(());
            }
            fs::remove_file(path)?;
        }

        let temp_path = path.with_extension("download");
        if temp_path.exists() {
            fs::remove_file(&temp_path)?;
        }

        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|error| {
                TruesightError::Embedding(format!("failed to build download client: {error}"))
            })?;

        let mut response = client
            .get(url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .map_err(|error| {
                TruesightError::Embedding(format!("failed to download {url}: {error}"))
            })?;

        let file = File::create(&temp_path)?;
        let mut writer = BufWriter::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 16 * 1024];

        loop {
            let read = response.read(&mut buffer).map_err(|error| {
                TruesightError::Embedding(format!("failed to read download stream: {error}"))
            })?;

            if read == 0 {
                break;
            }

            writer.write_all(&buffer[..read])?;
            hasher.update(&buffer[..read]);
        }

        writer.flush()?;

        let actual_sha256 = format!("{:x}", hasher.finalize());
        if actual_sha256 != expected_sha256 {
            let _ = fs::remove_file(&temp_path);
            return Err(TruesightError::Embedding(format!(
                "checksum mismatch for {}: expected {}, got {}",
                path.display(),
                expected_sha256,
                actual_sha256
            )));
        }

        fs::rename(temp_path, path)?;
        Ok(())
    }

    fn sha256_file(path: &Path) -> Result<String> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0_u8; 16 * 1024];

        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    fn encode_batch(&self, texts: &[&str]) -> Result<EncodedBatch> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|error| {
                TruesightError::Embedding(format!("failed to tokenize batch: {error}"))
            })?;

        let batch_size = encodings.len();
        let sequence_length = encodings
            .iter()
            .map(|encoding| encoding.get_ids().len())
            .max()
            .unwrap_or(0);
        let mut input_ids = Array2::<i64>::zeros((batch_size, sequence_length));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, sequence_length));
        let mut token_type_ids = Array2::<i64>::zeros((batch_size, sequence_length));

        for (row_index, encoding) in encodings.iter().enumerate() {
            for (column_index, token_id) in encoding.get_ids().iter().enumerate() {
                input_ids[(row_index, column_index)] = i64::from(*token_id);
            }

            for (column_index, mask_value) in encoding.get_attention_mask().iter().enumerate() {
                attention_mask[(row_index, column_index)] = i64::from(*mask_value);
            }

            for (column_index, token_type_id) in encoding.get_type_ids().iter().enumerate() {
                token_type_ids[(row_index, column_index)] = i64::from(*token_type_id);
            }
        }

        Ok(EncodedBatch {
            input_ids,
            attention_mask,
            token_type_ids,
        })
    }

    fn infer(&self, encoded: &EncodedBatch) -> Result<Vec<Vec<f32>>> {
        let input_ids = TensorRef::from_array_view(encoded.input_ids.view()).map_err(|error| {
            TruesightError::Embedding(format!("failed to prepare input_ids tensor: {error}"))
        })?;
        let attention_mask =
            TensorRef::from_array_view(encoded.attention_mask.view()).map_err(|error| {
                TruesightError::Embedding(format!(
                    "failed to prepare attention_mask tensor: {error}"
                ))
            })?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| TruesightError::Embedding("failed to lock ONNX session".to_string()))?;

        let outputs = if self.includes_token_type_ids {
            let token_type_ids = TensorRef::from_array_view(encoded.token_type_ids.view())
                .map_err(|error| {
                    TruesightError::Embedding(format!(
                        "failed to prepare token_type_ids tensor: {error}"
                    ))
                })?;

            session.run(ort::inputs![input_ids, attention_mask, token_type_ids])
        } else {
            session.run(ort::inputs![input_ids, attention_mask])
        }
        .map_err(|error| TruesightError::Embedding(format!("ONNX inference failed: {error}")))?;

        Self::extract_embeddings(outputs, encoded.attention_mask.view())
    }

    fn extract_embeddings(
        outputs: SessionOutputs<'_>,
        attention_mask: ArrayView2<'_, i64>,
    ) -> Result<Vec<Vec<f32>>> {
        for name in SENTENCE_OUTPUT_NAMES {
            if let Some(output) = outputs.get(*name) {
                let sentence_embeddings = output
                    .try_extract_array::<f32>()
                    .map_err(|error| {
                        TruesightError::Embedding(format!(
                            "failed to extract {name} tensor as f32 array: {error}"
                        ))
                    })?
                    .into_dimensionality::<Ix2>()
                    .map_err(|error| {
                        TruesightError::Embedding(format!(
                            "unexpected {name} tensor shape for sentence embeddings: {error}"
                        ))
                    })?;

                if sentence_embeddings.shape().get(1).copied() != Some(MODEL_DIMENSION) {
                    return Err(TruesightError::Embedding(format!(
                        "named output {name} has unexpected sentence embedding dimension {:?}; expected {MODEL_DIMENSION}",
                        sentence_embeddings.shape()
                    )));
                }

                return Ok(Self::normalize_embeddings(sentence_embeddings));
            }
        }

        for name in TOKEN_OUTPUT_NAMES {
            if let Some(output) = outputs.get(*name) {
                let token_embeddings = output
                    .try_extract_array::<f32>()
                    .map_err(|error| {
                        TruesightError::Embedding(format!(
                            "failed to extract {name} tensor as f32 array: {error}"
                        ))
                    })?
                    .into_dimensionality::<Ix3>()
                    .map_err(|error| {
                        TruesightError::Embedding(format!(
                            "unexpected {name} tensor shape for token embeddings: {error}"
                        ))
                    })?;

                if token_embeddings.shape().get(2).copied() != Some(MODEL_DIMENSION) {
                    return Err(TruesightError::Embedding(format!(
                        "named output {name} has unexpected token embedding dimension {:?}; expected {MODEL_DIMENSION}",
                        token_embeddings.shape()
                    )));
                }

                return Ok(Self::mean_pool_and_normalize(
                    token_embeddings,
                    attention_mask,
                ));
            }
        }

        let mut sentence_candidates = Vec::new();
        let mut token_candidates = Vec::new();

        for (name, output) in &outputs {
            let Ok(values) = output.try_extract_array::<f32>() else {
                continue;
            };

            if let Ok(sentence_embeddings) = values.view().into_dimensionality::<Ix2>() {
                if sentence_embeddings.shape().get(1).copied() == Some(MODEL_DIMENSION) {
                    sentence_candidates.push((name.to_string(), sentence_embeddings.to_owned()));
                    continue;
                }
            }

            if let Ok(token_embeddings) = values.view().into_dimensionality::<Ix3>() {
                if token_embeddings.shape().get(2).copied() == Some(MODEL_DIMENSION) {
                    token_candidates.push((name.to_string(), token_embeddings.to_owned()));
                }
            }
        }

        match sentence_candidates.len() {
            1 => {
                let (_, sentence_embeddings) = sentence_candidates.pop().expect("candidate exists");
                return Ok(Self::normalize_embeddings(sentence_embeddings.view()));
            }
            0 => {}
            _ => {
                return Err(TruesightError::Embedding(format!(
                    "multiple 2D embedding outputs matched MODEL_DIMENSION={MODEL_DIMENSION}: {}",
                    sentence_candidates
                        .iter()
                        .map(|(name, values)| format!("{name}{:?}", values.shape()))
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
        }

        match token_candidates.len() {
            1 => {
                let (_, token_embeddings) = token_candidates.pop().expect("candidate exists");
                return Ok(Self::mean_pool_and_normalize(
                    token_embeddings.view(),
                    attention_mask,
                ));
            }
            0 => {}
            _ => {
                return Err(TruesightError::Embedding(format!(
                    "multiple 3D embedding outputs matched MODEL_DIMENSION={MODEL_DIMENSION}: {}",
                    token_candidates
                        .iter()
                        .map(|(name, values)| format!("{name}{:?}", values.shape()))
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
        }

        Err(TruesightError::Embedding(format!(
            "unable to determine embedding output from ONNX model outputs: {}",
            Self::describe_outputs(&outputs)
        )))
    }

    fn describe_outputs(outputs: &SessionOutputs<'_>) -> String {
        outputs
            .iter()
            .map(|(name, output)| format!("{name}: {}", output.dtype()))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn normalize_embeddings(sentence_embeddings: ArrayView2<'_, f32>) -> Vec<Vec<f32>> {
        sentence_embeddings
            .axis_iter(Axis(0))
            .map(|embedding| {
                let mut values = embedding.to_vec();
                let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for value in &mut values {
                        *value /= norm;
                    }
                }
                values
            })
            .collect()
    }

    fn mean_pool_and_normalize(
        token_embeddings: ArrayView3<'_, f32>,
        attention_mask: ArrayView2<'_, i64>,
    ) -> Vec<Vec<f32>> {
        token_embeddings
            .axis_iter(Axis(0))
            .zip(attention_mask.axis_iter(Axis(0)))
            .map(|(sequence_embeddings, sequence_mask)| {
                let mut pooled = vec![0.0_f32; MODEL_DIMENSION];
                let mut token_count = 0.0_f32;

                for (token_embedding, mask_value) in sequence_embeddings
                    .axis_iter(Axis(0))
                    .zip(sequence_mask.iter())
                {
                    if *mask_value == 0 {
                        continue;
                    }

                    token_count += 1.0;
                    for (value, pooled_value) in token_embedding.iter().zip(pooled.iter_mut()) {
                        *pooled_value += *value;
                    }
                }

                if token_count > 0.0 {
                    for pooled_value in &mut pooled {
                        *pooled_value /= token_count;
                    }
                }

                let norm = pooled.iter().map(|value| value * value).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for pooled_value in &mut pooled {
                        *pooled_value /= norm;
                    }
                }

                pooled
            })
            .collect()
    }
}

impl Embedder for OnnxEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let batch = self.embed_batch(&[text])?;
        batch.into_iter().next().ok_or_else(|| {
            TruesightError::Embedding("embedder returned an empty batch".to_string())
        })
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
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
}

struct ModelAssets {
    model: PathBuf,
    tokenizer: PathBuf,
}

struct EncodedBatch {
    input_ids: Array2<i64>,
    attention_mask: Array2<i64>,
    token_type_ids: Array2<i64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, OnceLock};

    fn embedder() -> Arc<OnnxEmbedder> {
        static EMBEDDER: OnceLock<Arc<OnnxEmbedder>> = OnceLock::new();

        EMBEDDER
            .get_or_init(|| Arc::new(OnnxEmbedder::new().expect("embedder should initialize")))
            .clone()
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
}
