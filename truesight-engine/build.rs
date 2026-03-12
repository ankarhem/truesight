use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const REQUIRED_ORT_MINOR_VERSION: u32 = 23;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct OrtVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

fn main() {
    if !cfg!(target_os = "linux") {
        return;
    }

    println!("cargo:rerun-if-env-changed=ORT_DYLIB_PATH");

    match resolve_ort_dylib_path() {
        Ok(Some(path)) => {
            add_runtime_rpaths(Some(&path));
            println!("cargo:rustc-env=TRUESIGHT_ORT_DYLIB={}", path.display());
        }
        Ok(None) => add_runtime_rpaths(None),
        Err(error) => panic!("{error}"),
    }
}

fn add_runtime_rpaths(ort_dylib_path: Option<&Path>) {
    let Ok(entries) = fs::read_dir("/nix/store") else {
        return;
    };

    let mut runtime_dirs = entries
        .filter_map(|entry| entry.ok().map(|item| item.path()))
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let lib_dir = path.join("lib");
            let has_runtime_lib = (name.contains("gcc-")
                && name.ends_with("-lib")
                && lib_dir.join("libstdc++.so.6").exists())
                || (name.starts_with("openssl-") && lib_dir.join("libssl.so.3").exists());
            has_runtime_lib.then_some(lib_dir)
        })
        .collect::<Vec<_>>();

    if let Some(parent) = ort_dylib_path.and_then(Path::parent) {
        runtime_dirs.push(parent.to_path_buf());
    }

    runtime_dirs.sort();
    runtime_dirs.dedup();

    for dir in runtime_dirs {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}

fn resolve_ort_dylib_path() -> Result<Option<PathBuf>, String> {
    if let Some(path) = env::var_os("ORT_DYLIB_PATH") {
        return validate(path.into(), "ORT_DYLIB_PATH").map(Some);
    }

    let Ok(entries) = fs::read_dir("/nix/store") else {
        return Ok(None);
    };

    let mut candidates = entries
        .filter_map(|entry| entry.ok().map(|item| item.path()))
        .filter_map(|path| {
            let name = path.file_name()?.to_str()?;
            let lib_dir = path.join("lib");
            let dylib = lib_dir.join("libonnxruntime.so");

            if !name.contains("onnxruntime-") || !dylib.exists() {
                return None;
            }

            parse_ort_version(name).map(|version| (version, dylib))
        })
        .collect::<Vec<_>>();

    candidates.sort_by(|(left_version, left_path), (right_version, right_path)| {
        right_version
            .cmp(left_version)
            .then_with(|| left_path.cmp(right_path))
    });

    for (version, path) in candidates {
        if version.major == 1 && version.minor >= REQUIRED_ORT_MINOR_VERSION {
            return validate(path, "/nix/store").map(Some);
        }
    }

    Ok(None)
}

fn parse_ort_version(name: &str) -> Option<OrtVersion> {
    let marker = "onnxruntime-";
    let start = name.rfind(marker)? + marker.len();
    let version = name[start..].split('-').next()?;
    let mut parts = version.split('.');

    Some(OrtVersion {
        major: parts.next()?.parse().ok()?,
        minor: parts.next()?.parse().ok()?,
        patch: parts.next().unwrap_or("0").parse().ok()?,
    })
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
