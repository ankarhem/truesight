use ndarray::Array2;
use truesight_core::{Result, TruesightError};

use super::{EncodedBatch, OnnxEmbedder};

impl OnnxEmbedder {
    pub(super) fn encode_batch(&self, texts: &[&str]) -> Result<EncodedBatch> {
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
}
