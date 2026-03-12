use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=ORT_DYLIB_PATH");
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");

    if !cfg!(target_os = "linux") {
        return;
    }

    match resolve_ort_dylib_path() {
        Ok(Some(path)) => {
            add_runtime_rpath(path.parent());
            println!("cargo:rustc-env=TRUESIGHT_ORT_DYLIB={}", path.display());
        }
        Ok(None) => {}
        Err(error) => panic!("{error}"),
    }

    add_runtime_rpaths_from_env();
}

fn add_runtime_rpaths_from_env() {
    let Some(paths) = env::var_os("LD_LIBRARY_PATH") else {
        return;
    };

    let mut seen = Vec::new();
    for dir in env::split_paths(&paths) {
        if seen.iter().any(|seen_dir| seen_dir == &dir) {
            continue;
        }

        add_runtime_rpath(Some(dir.as_path()));
        seen.push(dir);
    }
}

fn add_runtime_rpath(dir: Option<&Path>) {
    let Some(dir) = dir else {
        return;
    };

    if dir.is_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}

fn resolve_ort_dylib_path() -> Result<Option<PathBuf>, String> {
    if let Some(path) = env::var_os("ORT_DYLIB_PATH") {
        return validate(path.into(), "ORT_DYLIB_PATH").map(Some);
    }

    Ok(None)
}

fn validate(path: PathBuf, source: &str) -> Result<PathBuf, String> {
    let canonical = path.canonicalize().map_err(|error| {
        format!(
            "failed to resolve ONNX Runtime dylib from {source} at {}: {error}",
            path.display()
        )
    })?;

    if canonical.is_file() {
        Ok(canonical)
    } else {
        Err(format!(
            "resolved ONNX Runtime dylib from {source} is not a file: {}",
            canonical.display()
        ))
    }
}
