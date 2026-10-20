// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Splitting a cleanup sweep into key ranges, each carrying only the references the
//! cleanup rules consult for keys in that range.
//!
//! Keys are dataset-relative object paths, compared byte by byte. The rules look up
//! a key itself, and for some keys a reference derived from it: a blob v2 sidecar
//! `data/{stem}/{blob}.blob` is kept by the data file `data/{stem}.lance`, and every
//! key below `_indices/{uuid}/` or `_fri/{map_id}/` by that uuid or map id. A
//! reference goes to every range holding a key it can decide.

use std::collections::HashSet;

use lance_core::{Error, Result};
use object_store::path::Path;
use serde::{Deserialize, Serialize};

use super::{CleanupSubtree, ReferencedFiles};

/// Known keys per task below which a subtree is not split further: a task that
/// lists less is not worth scheduling on its own.
pub(super) const KEYS_PER_TASK: usize = 10_000;

/// The keys `k` with `start_after < k <= end_inclusive`. A `None` bound is open.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct KeyRange {
    pub(super) start_after: Option<String>,
    pub(super) end_inclusive: Option<String>,
}

impl KeyRange {
    /// True if some key in `first..=last` is in this range.
    fn overlaps(&self, first: &str, last: &str) -> bool {
        self.start_after.as_deref().is_none_or(|start| last > start)
            && self.end_inclusive.as_deref().is_none_or(|end| first <= end)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReferenceKind {
    DataPath,
    ManagedBlob,
    Deletion,
    Transaction,
    IndexUuid,
    FragReuseMapId,
}

/// The keys whose cleanup decision can consult one reference.
struct Probe {
    subtree: CleanupSubtree,
    /// Every such key is in `first..=last`.
    first: String,
    last: String,
    /// Where a range boundary splits no group of keys sharing this reference.
    boundary: String,
}

const SUBTREES: [CleanupSubtree; 7] = [
    CleanupSubtree::Versions,
    CleanupSubtree::Transactions,
    CleanupSubtree::Data,
    CleanupSubtree::Blobs,
    CleanupSubtree::Indices,
    CleanupSubtree::Deletions,
    CleanupSubtree::FragReuse,
];

fn subtree_of(key: &str) -> Option<CleanupSubtree> {
    let dir = key.split('/').next()?;
    SUBTREES
        .into_iter()
        .find(|subtree| subtree.relative_dir() == dir)
}

/// `None` for a reference no listed key can consult.
fn probe(kind: ReferenceKind, entry: &str) -> Option<Probe> {
    let grouped = |subtree: CleanupSubtree, first: String, id: &str| {
        let dir = subtree.relative_dir();
        Probe {
            subtree,
            first,
            // `0` follows `/`, so this bounds every key below `{dir}/{id}/`.
            last: format!("{dir}/{id}0"),
            boundary: format!("{dir}/{id}"),
        }
    };
    match kind {
        ReferenceKind::IndexUuid | ReferenceKind::FragReuseMapId => {
            let subtree = if kind == ReferenceKind::IndexUuid {
                CleanupSubtree::Indices
            } else {
                CleanupSubtree::FragReuse
            };
            let first = format!("{}/{entry}", subtree.relative_dir());
            Some(grouped(subtree, first, entry))
        }
        ReferenceKind::DataPath => match entry
            .strip_prefix("data/")
            .and_then(|name| name.strip_suffix(".lance"))
        {
            // `.` sorts before `/`, so the data file is the first key of its group.
            Some(stem) if !stem.contains('/') => {
                Some(grouped(CleanupSubtree::Data, entry.to_string(), stem))
            }
            _ => exact(entry),
        },
        ReferenceKind::ManagedBlob | ReferenceKind::Deletion | ReferenceKind::Transaction => {
            exact(entry)
        }
    }
}

fn exact(key: &str) -> Option<Probe> {
    let subtree = subtree_of(key)?;
    // Keep a data file's sidecar directory with it: `data/{stem}` sorts before both
    // `data/{stem}.lance` and `data/{stem}/...`.
    let boundary = match (subtree, key.split('/').nth(1)) {
        (CleanupSubtree::Data, Some(name)) => {
            format!("data/{}", name.strip_suffix(".lance").unwrap_or(name))
        }
        _ => key.to_string(),
    };
    Some(Probe {
        subtree,
        first: key.to_string(),
        last: key.to_string(),
        boundary,
    })
}

/// The reference sets of [`ReferencedFiles`], as sorted dataset-relative paths;
/// index uuids and row-map ids are bare.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(super) struct ReferenceSlice {
    data_paths: Vec<String>,
    managed_blob_paths: Vec<String>,
    delete_paths: Vec<String>,
    tx_paths: Vec<String>,
    index_uuids: Vec<String>,
    frag_reuse_map_ids: Vec<String>,
}

impl ReferenceSlice {
    pub(super) fn from_files(files: &ReferencedFiles) -> Self {
        let paths = |set: &HashSet<Path>| sorted(set.iter().map(|path| path.to_string()));
        let ids = |set: &HashSet<String>| sorted(set.iter().cloned());
        Self {
            data_paths: paths(&files.data_paths),
            managed_blob_paths: paths(&files.managed_blob_paths),
            delete_paths: paths(&files.delete_paths),
            tx_paths: paths(&files.tx_paths),
            index_uuids: ids(&files.index_uuids),
            frag_reuse_map_ids: ids(&files.frag_reuse_map_ids),
        }
    }

    pub(super) fn to_files(&self) -> Result<ReferencedFiles> {
        let paths = |entries: &[String]| {
            entries
                .iter()
                .map(|entry| {
                    Path::parse(entry).map_err(|error| {
                        Error::invalid_input(format!(
                            "invalid path {entry:?} in a cleanup sweep task: {error}"
                        ))
                    })
                })
                .collect::<Result<HashSet<_>>>()
        };
        Ok(ReferencedFiles {
            data_paths: paths(&self.data_paths)?,
            managed_blob_paths: paths(&self.managed_blob_paths)?,
            delete_paths: paths(&self.delete_paths)?,
            tx_paths: paths(&self.tx_paths)?,
            index_uuids: self.index_uuids.iter().cloned().collect(),
            frag_reuse_map_ids: self.frag_reuse_map_ids.iter().cloned().collect(),
        })
    }

    pub(super) fn index_uuids(&self) -> &[String] {
        &self.index_uuids
    }

    pub(super) fn extend(&mut self, other: Self) {
        let Self {
            data_paths,
            managed_blob_paths,
            delete_paths,
            tx_paths,
            index_uuids,
            frag_reuse_map_ids,
        } = other;
        self.data_paths.extend(data_paths);
        self.managed_blob_paths.extend(managed_blob_paths);
        self.delete_paths.extend(delete_paths);
        self.tx_paths.extend(tx_paths);
        self.index_uuids.extend(index_uuids);
        self.frag_reuse_map_ids.extend(frag_reuse_map_ids);
    }

    /// The references that cleanup consults for keys of `subtree` in `range`.
    pub(super) fn restrict(&self, subtree: CleanupSubtree, range: &KeyRange) -> Self {
        let mut restricted = Self::default();
        for (kind, entry, probe) in self.probes(subtree) {
            if range.overlaps(&probe.first, &probe.last) {
                restricted.kind_mut(kind).push(entry.clone());
            }
        }
        restricted
    }

    fn kinds(&self) -> [(ReferenceKind, &[String]); 6] {
        [
            (ReferenceKind::DataPath, &self.data_paths),
            (ReferenceKind::ManagedBlob, &self.managed_blob_paths),
            (ReferenceKind::Deletion, &self.delete_paths),
            (ReferenceKind::Transaction, &self.tx_paths),
            (ReferenceKind::IndexUuid, &self.index_uuids),
            (ReferenceKind::FragReuseMapId, &self.frag_reuse_map_ids),
        ]
    }

    fn kind_mut(&mut self, kind: ReferenceKind) -> &mut Vec<String> {
        match kind {
            ReferenceKind::DataPath => &mut self.data_paths,
            ReferenceKind::ManagedBlob => &mut self.managed_blob_paths,
            ReferenceKind::Deletion => &mut self.delete_paths,
            ReferenceKind::Transaction => &mut self.tx_paths,
            ReferenceKind::IndexUuid => &mut self.index_uuids,
            ReferenceKind::FragReuseMapId => &mut self.frag_reuse_map_ids,
        }
    }

    /// The probes of the references whose keys are in `subtree`.
    fn probes(
        &self,
        subtree: CleanupSubtree,
    ) -> impl Iterator<Item = (ReferenceKind, &String, Probe)> {
        self.kinds().into_iter().flat_map(move |(kind, entries)| {
            entries.iter().filter_map(move |entry| {
                probe(kind, entry)
                    .filter(|probe| probe.subtree == subtree)
                    .map(|probe| (kind, entry, probe))
            })
        })
    }
}

fn sorted(entries: impl Iterator<Item = String>) -> Vec<String> {
    let mut entries = entries.collect::<Vec<_>>();
    entries.sort_unstable();
    entries
}

/// One key range of one subtree, with the references its keys consult.
#[derive(Debug)]
pub(super) struct Shard {
    pub(super) subtree: CleanupSubtree,
    pub(super) range: KeyRange,
    pub(super) referenced: ReferenceSlice,
    pub(super) verified: ReferenceSlice,
}

/// The sorted, distinct places where `subtree` can be split without separating
/// keys that consult the same reference. Their number is a proxy for the size of
/// the subtree.
pub(super) fn boundary_candidates(
    subtree: CleanupSubtree,
    referenced: &ReferenceSlice,
    verified: &ReferenceSlice,
) -> Vec<String> {
    let mut candidates = [referenced, verified]
        .into_iter()
        .flat_map(|slice| slice.probes(subtree))
        .map(|(_, _, probe)| probe.boundary)
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    candidates.dedup();
    candidates
}

/// Splits each subtree into ranges, giving every subtree at least one and the
/// rest of `max_tasks` to the subtree with the most known keys per range, as long
/// as each range keeps at least `keys_per_task` of them.
pub(super) fn allocate(known_keys: &[usize], max_tasks: usize, keys_per_task: usize) -> Vec<usize> {
    let caps = known_keys
        .iter()
        .map(|keys| keys.div_ceil(keys_per_task.max(1)).max(1))
        .collect::<Vec<_>>();
    let mut ranges = vec![1; known_keys.len()];
    let mut spare = max_tasks.saturating_sub(known_keys.len());
    while spare > 0 {
        // Ties go to the subtree listed first.
        let Some(next) = (0..ranges.len())
            .filter(|&i| ranges[i] < caps[i])
            .max_by(|&a, &b| {
                (known_keys[a] * ranges[b])
                    .cmp(&(known_keys[b] * ranges[a]))
                    .then(b.cmp(&a))
            })
        else {
            break;
        };
        ranges[next] += 1;
        spare -= 1;
    }
    ranges
}

/// Splits `subtree` into at most `num_ranges` ranges at quantiles of `candidates`,
/// from [`boundary_candidates`].
pub(super) fn split(
    subtree: CleanupSubtree,
    referenced: &ReferenceSlice,
    verified: &ReferenceSlice,
    candidates: &[String],
    num_ranges: usize,
) -> Vec<Shard> {
    let mut boundaries = (1..num_ranges)
        .filter_map(|i| candidates.get(candidates.len() * i / num_ranges))
        .collect::<Vec<_>>();
    boundaries.dedup();
    let starts = std::iter::once(None).chain(boundaries.iter().map(|b| Some((*b).clone())));
    let ends = boundaries
        .iter()
        .map(|b| Some((*b).clone()))
        .chain(std::iter::once(None));
    let mut shards = starts
        .zip(ends)
        .map(|(start_after, end_inclusive)| Shard {
            subtree,
            range: KeyRange {
                start_after,
                end_inclusive,
            },
            referenced: ReferenceSlice::default(),
            verified: ReferenceSlice::default(),
        })
        .collect::<Vec<_>>();
    // Range `i` holds the keys after `boundaries[i - 1]` up to `boundaries[i]`.
    let range_of = |key: &str| boundaries.partition_point(|boundary| boundary.as_str() < key);
    for (kind, entry, probe) in referenced.probes(subtree) {
        for shard in &mut shards[range_of(&probe.first)..=range_of(&probe.last)] {
            shard.referenced.kind_mut(kind).push(entry.clone());
        }
    }
    for (kind, entry, probe) in verified.probes(subtree) {
        for shard in &mut shards[range_of(&probe.first)..=range_of(&probe.last)] {
            shard.verified.kind_mut(kind).push(entry.clone());
        }
    }
    shards
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::cleanup::{CleanupInspection, CleanupTask};
    use chrono::{DateTime, Utc};
    use object_store::ObjectMeta;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    fn random_hex(rng: &mut StdRng, len: usize) -> String {
        (0..len)
            .map(|_| char::from_digit(rng.random_range(0..16), 16).unwrap())
            .collect()
    }

    fn random_uuid(rng: &mut StdRng) -> String {
        uuid::Uuid::from_u128(rng.random()).to_string()
    }

    /// A data file stem in the current layout (binary prefix) or the legacy one.
    fn random_stem(rng: &mut StdRng) -> String {
        if rng.random_bool(0.7) {
            let bits = (0..24)
                .map(|_| if rng.random_bool(0.5) { '1' } else { '0' })
                .collect::<String>();
            format!("{bits}{}", random_hex(rng, 26))
        } else {
            random_uuid(rng)
        }
    }

    /// Keys a listing of `subtree` could return, and references to a random subset
    /// of them, split into referenced and verified.
    fn fixture(
        rng: &mut StdRng,
        subtree: CleanupSubtree,
    ) -> (Vec<String>, ReferencedFiles, ReferencedFiles) {
        let mut keys = Vec::new();
        let mut referenced = ReferencedFiles::default();
        let mut verified = ReferencedFiles::default();
        for _ in 0..200 {
            let target = if rng.random_bool(0.5) {
                &mut referenced
            } else {
                &mut verified
            };
            let is_known = rng.random_bool(0.6);
            let mut exact_key = |key: String, set: &mut HashSet<Path>| {
                if is_known {
                    set.insert(Path::from(key.as_str()));
                }
                keys.push(key);
            };
            match subtree {
                CleanupSubtree::Data => {
                    let stem = random_stem(rng);
                    let extra = rng.random_range(0..3);
                    let tmp = rng.random_bool(0.2);
                    let bare = rng.random_bool(0.1);
                    exact_key(format!("data/{stem}.lance"), &mut target.data_paths);
                    for _ in 0..extra {
                        keys.push(format!("data/{stem}/{}.blob", random_hex(rng, 32)));
                    }
                    if tmp {
                        keys.push(format!("data/{stem}/.tmpAb3Xz9"));
                    }
                    if bare {
                        keys.push(format!("data/{stem}"));
                    }
                }
                CleanupSubtree::Indices | CleanupSubtree::FragReuse => {
                    let id = random_uuid(rng);
                    let dir = subtree.relative_dir();
                    for name in ["index.idx", "auxiliary.idx", "part/0.lance"] {
                        keys.push(format!("{dir}/{id}/{name}"));
                    }
                    if is_known {
                        if subtree == CleanupSubtree::Indices {
                            target.index_uuids.insert(id);
                        } else {
                            target.frag_reuse_map_ids.insert(id);
                        }
                    }
                }
                CleanupSubtree::Deletions => {
                    let key = format!(
                        "_deletions/{}-{}-{}.arrow",
                        rng.random_range(0..2000),
                        rng.random_range(0..20),
                        rng.random::<u64>()
                    );
                    exact_key(key, &mut target.delete_paths);
                }
                CleanupSubtree::Transactions => {
                    let key = format!(
                        "_transactions/{}-{}.txn",
                        rng.random_range(0..2000),
                        random_uuid(rng)
                    );
                    exact_key(key, &mut target.tx_paths);
                }
                CleanupSubtree::Blobs => {
                    let key = format!("_blobs/{}.blob", random_uuid(rng));
                    exact_key(key, &mut target.managed_blob_paths);
                }
                CleanupSubtree::Versions => {
                    keys.push(format!("_versions/.tmp{}", random_hex(rng, 8)));
                }
            }
        }
        // Managed blob objects can also live in a data file's sidecar directory.
        if subtree == CleanupSubtree::Data {
            for key in keys.iter().filter(|key| key.ends_with(".blob")).step_by(3) {
                verified.managed_blob_paths.insert(Path::from(key.as_str()));
            }
        }
        keys.sort();
        keys.dedup();
        (keys, referenced, verified)
    }

    fn classify(
        key: &str,
        referenced: &ReferencedFiles,
        verified: &ReferencedFiles,
        maybe_in_progress: bool,
    ) -> Option<(String, bool)> {
        let inspection = CleanupInspection {
            referenced_files: referenced.clone(),
            verified_files: verified.clone(),
            ..Default::default()
        };
        let meta = ObjectMeta {
            location: Path::from(key),
            last_modified: DateTime::<Utc>::UNIX_EPOCH,
            size: 1,
            e_tag: None,
            version: None,
        };
        CleanupTask::cleanup_file_if_not_referenced(
            &Path::default(),
            meta,
            maybe_in_progress,
            &inspection,
        )
        .unwrap()
        .map(|file| (file.path.to_string(), file.unverified))
    }

    fn contains(range: &KeyRange, key: &str) -> bool {
        range.overlaps(key, key)
    }

    #[rstest]
    #[case::versions(CleanupSubtree::Versions)]
    #[case::transactions(CleanupSubtree::Transactions)]
    #[case::data(CleanupSubtree::Data)]
    #[case::blobs(CleanupSubtree::Blobs)]
    #[case::indices(CleanupSubtree::Indices)]
    #[case::deletions(CleanupSubtree::Deletions)]
    #[case::frag_reuse(CleanupSubtree::FragReuse)]
    fn shards_classify_like_the_whole_subtree(
        #[case] subtree: CleanupSubtree,
        #[values(1, 2, 7, 64)] num_ranges: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(num_ranges as u64);
        let (keys, referenced, verified) = fixture(&mut rng, subtree);
        let referenced_slice = ReferenceSlice::from_files(&referenced);
        let verified_slice = ReferenceSlice::from_files(&verified);
        let candidates = boundary_candidates(subtree, &referenced_slice, &verified_slice);
        let shards = split(
            subtree,
            &referenced_slice,
            &verified_slice,
            &candidates,
            num_ranges,
        );
        assert!(!shards.is_empty() && shards.len() <= num_ranges);
        if num_ranges > 1 && subtree != CleanupSubtree::Versions {
            assert!(shards.len() > 1, "{subtree}: {} ranges", shards.len());
        }

        // Keys equal to a boundary are in the listing as well.
        let mut all_keys = keys.clone();
        all_keys.extend(
            shards
                .iter()
                .filter_map(|shard| shard.range.end_inclusive.clone()),
        );
        for key in &all_keys {
            let owners = shards
                .iter()
                .filter(|shard| contains(&shard.range, key))
                .collect::<Vec<_>>();
            assert_eq!(owners.len(), 1, "{key} is in {} ranges", owners.len());
            let shard_referenced = owners[0].referenced.to_files().unwrap();
            let shard_verified = owners[0].verified.to_files().unwrap();
            for maybe_in_progress in [false, true] {
                assert_eq!(
                    classify(key, &shard_referenced, &shard_verified, maybe_in_progress),
                    classify(key, &referenced, &verified, maybe_in_progress),
                    "{key}, maybe_in_progress={maybe_in_progress}"
                );
            }
        }

        // A data file and its sidecars are listed by the same task.
        if subtree == CleanupSubtree::Data {
            let owner = |key: &str| shards.iter().position(|shard| contains(&shard.range, key));
            for key in keys.iter().filter(|key| key.ends_with(".blob")) {
                let data_file = format!("{}.lance", key.rsplit_once('/').unwrap().0);
                assert_eq!(owner(key), owner(&data_file), "{key}");
            }
        }

        // `split` and `restrict` agree on which references a range needs.
        for shard in &shards {
            assert_eq!(
                shard.referenced,
                referenced_slice.restrict(subtree, &shard.range)
            );
            assert_eq!(
                shard.verified,
                verified_slice.restrict(subtree, &shard.range)
            );
        }
    }

    #[test]
    fn references_go_only_to_their_subtree() {
        let files = ReferencedFiles {
            data_paths: HashSet::from([Path::from("data/a.lance")]),
            managed_blob_paths: HashSet::from([Path::from("_blobs/b.blob")]),
            delete_paths: HashSet::from([Path::from("_deletions/1-2-3.arrow")]),
            tx_paths: HashSet::from([Path::from("_transactions/1-x.txn")]),
            index_uuids: HashSet::from(["u".to_string()]),
            frag_reuse_map_ids: HashSet::from(["m".to_string()]),
        };
        let slice = ReferenceSlice::from_files(&files);
        assert_eq!(slice.to_files().unwrap().data_paths, files.data_paths);
        for subtree in SUBTREES {
            let restricted = slice.restrict(subtree, &KeyRange::default());
            let count = restricted
                .kinds()
                .iter()
                .map(|(_, entries)| entries.len())
                .sum::<usize>();
            let expected = usize::from(subtree != CleanupSubtree::Versions);
            assert_eq!(count, expected, "{subtree}");
        }
    }

    #[rstest]
    #[case::one_per_subtree(&[0, 0, 0], 1, &[1, 1, 1])]
    #[case::capped_by_known_keys(&[5, 25_000, 0], 64, &[1, 3, 1])]
    #[case::proportional(&[100_000, 300_000], 9, &[3, 6])]
    #[case::no_spare(&[100_000, 300_000], 2, &[1, 1])]
    fn allocates_ranges(
        #[case] known_keys: &[usize],
        #[case] max_tasks: usize,
        #[case] expected: &[usize],
    ) {
        assert_eq!(allocate(known_keys, max_tasks, KEYS_PER_TASK), expected);
    }
}
