use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use truesight_core::{Result, TruesightError};

use super::{MODEL_NAME, MODEL_SHA256, MODEL_URL, OnnxEmbedder, TOKENIZER_SHA256, TOKENIZER_URL};

pub(super) struct ModelAssets {
    pub(super) model: PathBuf,
    pub(super) tokenizer: PathBuf,
}

impl OnnxEmbedder {
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

    pub(super) fn ensure_model() -> Result<ModelAssets> {
        std::thread::scope(|s| {
            let model_dir = Self::model_dir()?;
            fs::create_dir_all(&model_dir)?;

            let model = model_dir.join("model.onnx");
            let tokenizer = model_dir.join("tokenizer.json");

            let model_path = model.clone();
            let tokenizer_path = tokenizer.clone();

            let model_task = s.spawn(move || {
                Self::ensure_file(&model_path, MODEL_URL, MODEL_SHA256)?;
                Ok::<_, TruesightError>(())
            });
            let tokenizer_task = s.spawn(move || {
                Self::ensure_file(&tokenizer_path, TOKENIZER_URL, TOKENIZER_SHA256)?;
                Ok::<_, TruesightError>(())
            });

            model_task.join().unwrap_or_else(|_| {
                Err(TruesightError::Embedding(
                    "model download thread panicked".into(),
                ))
            })?;
            tokenizer_task.join().unwrap_or_else(|_| {
                Err(TruesightError::Embedding(
                    "tokenizer download thread panicked".into(),
                ))
            })?;

            Ok(ModelAssets { model, tokenizer })
        })
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
}
