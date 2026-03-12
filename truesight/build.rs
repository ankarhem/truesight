use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");

    if !cfg!(target_os = "linux") {
        return;
    }

    let Some(paths) = env::var_os("LD_LIBRARY_PATH") else {
        return;
    };

    let mut seen = Vec::new();
    for dir in env::split_paths(&paths) {
        if seen.iter().any(|seen_dir| seen_dir == &dir) {
            continue;
        }

        add_runtime_rpath(dir.as_path());
        seen.push(dir);
    }
}

fn add_runtime_rpath(dir: &Path) {
    if dir.is_dir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}
