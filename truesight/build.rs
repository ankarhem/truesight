use std::fs;
use std::path::PathBuf;

fn main() {
    if !cfg!(target_os = "linux") {
        return;
    }

    let Ok(entries) = fs::read_dir("/nix/store") else {
        return;
    };

    let mut runtime_dirs = entries
        .filter_map(|entry| entry.ok().map(|item| item.path()))
        .filter_map(runtime_lib_dir)
        .collect::<Vec<_>>();

    runtime_dirs.sort();
    runtime_dirs.dedup();

    for dir in runtime_dirs {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}

fn runtime_lib_dir(path: PathBuf) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let lib_dir = path.join("lib");

    let has_runtime_lib = (name.contains("gcc-")
        && name.ends_with("-lib")
        && lib_dir.join("libstdc++.so.6").exists())
        || (name.starts_with("openssl-") && lib_dir.join("libssl.so.3").exists());

    has_runtime_lib.then_some(lib_dir)
}
