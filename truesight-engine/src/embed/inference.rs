use ndarray::{ArrayView2, ArrayView3, Axis, Ix2, Ix3};
use ort::{session::SessionOutputs, value::TensorRef};
use truesight_core::{Result, TruesightError};

use super::{
    EncodedBatch, MODEL_DIMENSION, OnnxEmbedder, SENTENCE_OUTPUT_NAMES, TOKEN_OUTPUT_NAMES,
};

impl OnnxEmbedder {
    pub(super) fn infer(&self, encoded: &EncodedBatch) -> Result<Vec<Vec<f32>>> {
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
