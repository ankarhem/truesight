use std::path::{Path, PathBuf};

pub(super) fn normalized_filter(filter: Option<&str>) -> Option<String> {
    filter
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.trim_matches('/').replace('\\', "/"))
        .filter(|value| !value.is_empty() && value != ".")
}

pub(super) fn matches_filter(root: &Path, file_path: &Path, filter: Option<&str>) -> bool {
    let Some(filter) = filter else {
        return true;
    };

    let relative = relative_path(root, file_path)
        .to_string_lossy()
        .replace('\\', "/");
    relative == filter || relative.starts_with(&format!("{filter}/"))
}

pub(super) fn relative_path(root: &Path, file_path: &Path) -> PathBuf {
    if file_path.is_absolute() {
        file_path
            .strip_prefix(root)
            .map(Path::to_path_buf)
            .unwrap_or_else(|_| file_path.to_path_buf())
    } else {
        file_path.to_path_buf()
    }
}

pub(super) fn module_path(file_path: &Path) -> PathBuf {
    file_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub(super) fn module_name(root: &Path, current_module_path: &Path) -> String {
    if current_module_path == Path::new(".") {
        root.file_name()
            .and_then(|segment| segment.to_str())
            .filter(|segment| !segment.is_empty())
            .unwrap_or("root")
            .to_string()
    } else {
        current_module_path
            .file_name()
            .and_then(|segment| segment.to_str())
            .filter(|segment| !segment.is_empty())
            .unwrap_or("root")
            .to_string()
    }
}

pub(super) fn module_label(root: &Path, current_module_path: &Path) -> String {
    if current_module_path == Path::new(".") {
        module_name(root, current_module_path)
    } else {
        current_module_path.to_string_lossy().replace('\\', "/")
    }
}

pub(super) fn file_name_for_symbol(file_path: &Path) -> String {
    file_path
        .file_name()
        .and_then(|segment| segment.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| file_path.to_string_lossy().replace('\\', "/"))
}
