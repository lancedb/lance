// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! The manifest regions an operation writes, and whether two footprints collide.
//!
//! Conflict detection in the action vocabulary is a *footprint* comparison rather
//! than a pairwise match: every action and every legacy [`Operation`] declares the
//! regions of the manifest it writes, and two transactions conflict exactly when
//! their footprints intersect. The intersection is computed once, here, so the
//! cost of a new action is one declaration instead of a row and a column in an
//! N-by-N table.
//!
//! [`Operation`]: crate::transaction::Operation

use std::collections::{BTreeMap, BTreeSet};

/// A region of the manifest an operation can write.
///
/// Deliberately coarse. Over-declaring a region costs a spurious retry, but an
/// *unlisted* region is a silently permitted conflict, so every region is defined
/// here before any action needs it and refined to key granularity per slice.
///
/// Id counters are not regions and never will be. Every operation that needs an
/// id is expressed as *adding to* the counter, and additions to a monotone
/// counter cannot conflict: each side mints from whatever the counter holds when
/// it applies, so two concurrent minting operations get disjoint ranges by
/// construction. Making a counter a region would manufacture exactly the
/// conflicts the minting design exists to avoid — it is why two `Append`s
/// commute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Region {
    /// `Manifest::base_paths`.
    Bases,
    /// The field structure of `Manifest::schema`.
    Schema,
    /// The metadata maps on `Manifest::schema` and its fields.
    SchemaMetadata,
    /// `Manifest::fragments`, including their data files and deletion files.
    Fragments,
    /// The index metadata list committed alongside the manifest.
    Indices,
    /// `Manifest::config`.
    Config,
    /// `Manifest::table_metadata`.
    TableMetadata,
    /// MemWAL bookkeeping: which SSTable generations are compacted into the
    /// base table.
    Generations,
}

impl Region {
    /// Every region. Used to build the footprint of operations that replace the
    /// manifest wholesale.
    pub const ALL: [Self; 8] = [
        Self::Bases,
        Self::Schema,
        Self::SchemaMetadata,
        Self::Fragments,
        Self::Indices,
        Self::Config,
        Self::TableMetadata,
        Self::Generations,
    ];
}

/// A named claim inside a region, for regions where an operation can say more
/// than "I touch this".
///
/// Keys are compared only within one region, so the same string under two
/// regions is two unrelated claims.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Key {
    /// A base path's user-visible name (`BasePath::name`).
    Name(String),
    /// A base path's full URI (`BasePath::path`).
    Path(String),
    /// An entry in one of the manifest's string maps.
    Entry(String),
}

/// How much of a region an operation writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Scope {
    /// The whole region. Intersects any other claim on it.
    All,
    /// Only these named claims.
    Keys(BTreeSet<Key>),
}

/// Whether two intersecting footprints can be resolved by retrying the commit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conflict {
    /// Re-applying against the newer manifest may succeed.
    Retryable,
    /// No retry can succeed, because both sides claim the same name.
    Incompatible,
}

/// The set of manifest regions an operation writes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ManifestMask {
    regions: BTreeMap<Region, Scope>,
}

impl ManifestMask {
    /// Every region, in full — the footprint of an operation that replaces the
    /// manifest wholesale.
    pub fn everything() -> Self {
        Self {
            regions: Region::ALL.iter().map(|r| (*r, Scope::All)).collect(),
        }
    }

    /// Claim a whole region.
    pub fn with_all(mut self, region: Region) -> Self {
        self.regions.insert(region, Scope::All);
        self
    }

    /// Claim named keys within a region, merging with any claim already made on
    /// it. A region already claimed in full stays claimed in full.
    pub fn with_keys(mut self, region: Region, keys: impl IntoIterator<Item = Key>) -> Self {
        match self
            .regions
            .entry(region)
            .or_insert_with(|| Scope::Keys(BTreeSet::new()))
        {
            Scope::All => {}
            Scope::Keys(existing) => existing.extend(keys),
        }
        self
    }

    pub fn union(mut self, other: Self) -> Self {
        for (region, scope) in other.regions {
            match scope {
                Scope::All => self = self.with_all(region),
                Scope::Keys(keys) => self = self.with_keys(region, keys),
            }
        }
        self
    }

    /// Whether this footprint touches any of `regions`, at any granularity.
    pub fn intersects_any(&self, regions: &[Region]) -> bool {
        regions
            .iter()
            .any(|region| self.regions.contains_key(region))
    }

    /// The verdict for two footprints, or `None` if they are disjoint.
    ///
    /// Exact-key overlap is incompatible: key-level claims are exclusive, and no
    /// retry frees a base name or a config key that another commit has taken.
    /// Any coarser overlap is retryable, because a retry re-runs apply against
    /// the newer manifest and apply re-validates.
    pub fn conflicts_with(&self, other: &Self) -> Option<Conflict> {
        let mut verdict = None;
        for (region, scope) in &self.regions {
            let Some(other_scope) = other.regions.get(region) else {
                continue;
            };
            match (scope, other_scope) {
                (Scope::Keys(ours), Scope::Keys(theirs)) => {
                    if !ours.is_disjoint(theirs) {
                        return Some(Conflict::Incompatible);
                    }
                }
                _ => verdict = Some(Conflict::Retryable),
            }
        }
        verdict
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys(region: Region, keys: impl IntoIterator<Item = Key>) -> ManifestMask {
        ManifestMask::default().with_keys(region, keys)
    }

    #[test]
    fn test_disjoint_regions_do_not_conflict() {
        let bases = ManifestMask::default().with_all(Region::Bases);
        let fragments = ManifestMask::default().with_all(Region::Fragments);
        assert_eq!(bases.conflicts_with(&fragments), None);
        assert_eq!(fragments.conflicts_with(&bases), None);
    }

    #[test]
    fn test_distinct_keys_in_one_region_do_not_conflict() {
        let warm = keys(Region::Bases, [Key::Name("warm".into())]);
        let cold = keys(Region::Bases, [Key::Name("cold".into())]);
        assert_eq!(warm.conflicts_with(&cold), None);
    }

    #[test]
    fn test_exact_key_overlap_is_incompatible() {
        let warm = keys(Region::Bases, [Key::Name("warm".into())]);
        assert_eq!(
            warm.conflicts_with(&warm.clone()),
            Some(Conflict::Incompatible)
        );
    }

    #[test]
    fn test_key_variants_are_distinct_claims() {
        // A base *named* "s3://bucket/a" does not claim the base *at* that path.
        let by_name = keys(Region::Bases, [Key::Name("s3://bucket/a".into())]);
        let by_path = keys(Region::Bases, [Key::Path("s3://bucket/a".into())]);
        assert_eq!(by_name.conflicts_with(&by_path), None);
    }

    #[test]
    fn test_same_key_string_in_different_regions_does_not_conflict() {
        let config = keys(Region::Config, [Key::Entry("k".into())]);
        let table_metadata = keys(Region::TableMetadata, [Key::Entry("k".into())]);
        assert_eq!(config.conflicts_with(&table_metadata), None);
    }

    #[test]
    fn test_coarse_overlap_is_retryable() {
        let all_bases = ManifestMask::default().with_all(Region::Bases);
        let one_base = keys(Region::Bases, [Key::Name("warm".into())]);
        assert_eq!(
            all_bases.conflicts_with(&one_base),
            Some(Conflict::Retryable)
        );
        assert_eq!(
            one_base.conflicts_with(&all_bases),
            Some(Conflict::Retryable)
        );
        assert_eq!(
            all_bases.conflicts_with(&all_bases.clone()),
            Some(Conflict::Retryable)
        );
    }

    #[test]
    fn test_incompatible_wins_over_retryable() {
        // Overlapping keys in one region and coarse overlap in another: the
        // stricter verdict must win regardless of region iteration order.
        let a = keys(Region::Bases, [Key::Name("warm".into())]).with_all(Region::Fragments);
        let b = keys(Region::Bases, [Key::Name("warm".into())]).with_all(Region::Fragments);
        assert_eq!(a.conflicts_with(&b), Some(Conflict::Incompatible));
    }

    #[test]
    fn test_everything_conflicts_with_any_claim() {
        for region in Region::ALL {
            let mask = ManifestMask::default().with_all(region);
            assert!(
                ManifestMask::everything().conflicts_with(&mask).is_some(),
                "everything() must cover {region:?}"
            );
        }
    }

    #[test]
    fn test_with_all_subsumes_keys() {
        let mask = keys(Region::Bases, [Key::Name("warm".into())]).with_all(Region::Bases);
        // Now coarse, so an unrelated name still overlaps — and retryably, not
        // as an exact-key claim.
        let other = keys(Region::Bases, [Key::Name("cold".into())]);
        assert_eq!(mask.conflicts_with(&other), Some(Conflict::Retryable));
    }

    #[test]
    fn test_with_keys_does_not_downgrade_all() {
        let mask = ManifestMask::default()
            .with_all(Region::Bases)
            .with_keys(Region::Bases, [Key::Name("warm".into())]);
        let other = keys(Region::Bases, [Key::Name("cold".into())]);
        assert_eq!(mask.conflicts_with(&other), Some(Conflict::Retryable));
    }

    #[test]
    fn test_union_merges_keys_and_regions() {
        let a = keys(Region::Bases, [Key::Name("warm".into())]);
        let b = keys(Region::Bases, [Key::Name("cold".into())]).with_all(Region::Fragments);
        let merged = a.union(b);

        assert_eq!(
            merged.conflicts_with(&keys(Region::Bases, [Key::Name("warm".into())])),
            Some(Conflict::Incompatible)
        );
        assert_eq!(
            merged.conflicts_with(&keys(Region::Bases, [Key::Name("cold".into())])),
            Some(Conflict::Incompatible)
        );
        assert!(merged.intersects_any(&[Region::Fragments]));
    }

    #[test]
    fn test_intersects_any() {
        let mask = keys(Region::Bases, [Key::Name("warm".into())]);
        assert!(mask.intersects_any(&[Region::Fragments, Region::Bases]));
        assert!(!mask.intersects_any(&[Region::Fragments, Region::Indices]));
        assert!(!mask.intersects_any(&[]));
    }
}
