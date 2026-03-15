use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use rayon::prelude::*;
use truesight_core::{IncrementalStorage, Result};

use super::ChangeSet;
use crate::indexing::repo_relative_path;
use crate::util::hash_bytes;
use crate::walker::FileWalker;

pub(super) fn detect_initial_changes(walker: &FileWalker, root: &Path) -> Result<ChangeSet> {
    let mut changes = ChangeSet {
        added: walker
            .walk(root)?
            .into_iter()
            .map(|file| repo_relative_path(root, &file.path))
            .collect::<Result<Vec<_>>>()?,
        modified: Vec::new(),
        deleted: Vec::new(),
    };
    changes.normalize();
    Ok(changes)
}

pub(super) async fn detect_hash_changes<S: IncrementalStorage + ?Sized>(
    walker: &FileWalker,
    root: &Path,
    storage: &S,
    repo_id: &str,
    branch: &str,
) -> Result<ChangeSet> {
    let indexed = storage.get_indexed_files(repo_id, branch).await?;
    let indexed_by_path = indexed
        .into_iter()
        .map(|record| (record.file_path, record.file_hash))
        .collect::<HashMap<_, _>>();
    let current_hashes = walker
        .walk(root)?
        .into_par_iter()
        .map(|discovered| {
            let relative_path = repo_relative_path(root, &discovered.path)?;
            let file_hash = hash_file(&discovered.path)?;
            Ok((relative_path, file_hash))
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<HashMap<_, _>>>()?;

    let mut changes = ChangeSet::default();

    for (path, current_hash) in &current_hashes {
        match indexed_by_path.get(path) {
            None => changes.added.push(path.clone()),
            Some(stored_hash) if stored_hash != current_hash => changes.modified.push(path.clone()),
            Some(_) => {}
        }
    }

    let current_paths = current_hashes.keys().cloned().collect::<HashSet<_>>();
    for path in indexed_by_path.keys() {
        if !current_paths.contains(path) {
            changes.deleted.push(path.clone());
        }
    }

    changes.normalize();
    Ok(changes)
}

fn hash_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(hash_bytes(&bytes))
}
