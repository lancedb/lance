// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Reproducible comparison of the legacy per-row FRI maps and the compact
//! bitmap/rank representation.
//!
//! This benchmark starts from the same decoded `FragReuseIndexDetails` for
//! both implementations. The open measurement includes Roaring deserialization
//! and construction of the queryable runtime representation, but excludes
//! object-store I/O and protobuf decoding.

#![allow(clippy::print_stdout)]

use std::collections::HashMap;
use std::hint::black_box;
use std::io::Cursor;
use std::time::Instant;

use lance::dataset::optimize::remapping::transpose_row_ids_from_digest;
use lance_core::deepsize::{Context, DeepSizeOf};
use lance_core::utils::address::RowAddress;
use lance_index::frag_reuse::{
    CompactFragReuseIndex, FragDigest, FragReuseGroup, FragReuseIndexDetails, FragReuseVersion,
};
use roaring::RoaringTreemap;
use serde_json::json;
use uuid::Uuid;

#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    rows: usize,
    changed_basis_points: u32,
    chain_len: usize,
    old_fragment_count: usize,
    groups_per_version: usize,
}

struct Config {
    repeats: usize,
    lookups: usize,
    batch_size: usize,
    quick: bool,
}

struct LegacyFragReuseIndex {
    row_id_maps: Vec<HashMap<u64, Option<u64>>>,
    details: FragReuseIndexDetails,
}

impl LegacyFragReuseIndex {
    fn open(details: &FragReuseIndexDetails) -> Self {
        let mut row_id_maps = Vec::with_capacity(details.versions.len());
        for version in &details.versions {
            let mut row_id_map = HashMap::new();
            for group in &version.groups {
                let changed_row_addrs =
                    RoaringTreemap::deserialize_from(Cursor::new(&group.changed_row_addrs))
                        .unwrap();
                row_id_map.extend(transpose_row_ids_from_digest(
                    changed_row_addrs,
                    &group.old_frags,
                    &group.new_frags,
                ));
            }
            row_id_maps.push(row_id_map);
        }
        Self {
            row_id_maps,
            details: details.clone(),
        }
    }

    fn remap_row_id(&self, row_id: u64) -> Option<u64> {
        let mut mapped = Some(row_id);
        for row_id_map in &self.row_id_maps {
            if let Some(current) = mapped {
                mapped = row_id_map.get(&current).copied().unwrap_or(mapped);
            }
        }
        mapped
    }

    fn remap_row_ids_in_place(&self, row_ids: &mut [Option<u64>]) {
        for row_id in row_ids {
            if let Some(current) = *row_id {
                *row_id = self.remap_row_id(current);
            }
        }
    }
}

impl DeepSizeOf for LegacyFragReuseIndex {
    fn deep_size_of_children(&self, context: &mut Context) -> usize {
        self.row_id_maps.deep_size_of_children(context)
            + self.details.deep_size_of_children(context)
    }
}

fn main() {
    let config = parse_config();
    let cases = if config.quick {
        vec![Case {
            name: "quick",
            rows: 100_000,
            changed_basis_points: 5_000,
            chain_len: 2,
            old_fragment_count: 16,
            groups_per_version: 2,
        }]
    } else {
        vec![
            Case {
                name: "small_very_sparse",
                rows: 100_000,
                changed_basis_points: 10,
                chain_len: 1,
                old_fragment_count: 16,
                groups_per_version: 1,
            },
            Case {
                name: "small_sparse",
                rows: 100_000,
                changed_basis_points: 100,
                chain_len: 1,
                old_fragment_count: 16,
                groups_per_version: 1,
            },
            Case {
                name: "medium_sparse_chain",
                rows: 1_000_000,
                changed_basis_points: 1_000,
                chain_len: 4,
                old_fragment_count: 64,
                groups_per_version: 8,
            },
            Case {
                name: "medium_balanced",
                rows: 1_000_000,
                changed_basis_points: 5_000,
                chain_len: 1,
                old_fragment_count: 64,
                groups_per_version: 8,
            },
            Case {
                name: "medium_dense",
                rows: 1_000_000,
                changed_basis_points: 9_000,
                chain_len: 1,
                old_fragment_count: 64,
                groups_per_version: 8,
            },
            Case {
                name: "fragmented_sparse",
                rows: 1_000_000,
                changed_basis_points: 100,
                chain_len: 1,
                old_fragment_count: 4_096,
                groups_per_version: 256,
            },
            Case {
                name: "fragmented_dense_chain",
                rows: 500_000,
                changed_basis_points: 9_000,
                chain_len: 4,
                old_fragment_count: 1_024,
                groups_per_version: 128,
            },
            Case {
                name: "long_dense_chain",
                rows: 250_000,
                changed_basis_points: 9_000,
                chain_len: 8,
                old_fragment_count: 128,
                groups_per_version: 16,
            },
            Case {
                name: "large_sparse",
                rows: 10_000_000,
                changed_basis_points: 100,
                chain_len: 1,
                old_fragment_count: 128,
                groups_per_version: 16,
            },
            Case {
                name: "large_balanced_chain",
                rows: 5_000_000,
                changed_basis_points: 5_000,
                chain_len: 4,
                old_fragment_count: 256,
                groups_per_version: 32,
            },
        ]
    };

    println!(
        "{}",
        json!({
            "type": "environment",
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "repeats": config.repeats,
            "lookups": config.lookups,
            "batch_size": config.batch_size,
            "unaffected_query_percent": 10,
            "profile": "cargo bench (release)",
            "memory_metric": "DeepSizeOf retained-bytes proxy; Roaring containers use serialized_size",
        })
    );

    for case in cases {
        run_case(case, &config);
    }
}

fn parse_config() -> Config {
    let mut config = Config {
        repeats: 30,
        lookups: 200_000,
        batch_size: 65_536,
        quick: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            // `cargo bench` passes this libtest-compatible marker even for a
            // custom harness.
            "--bench" => {}
            "--quick" => {
                config.quick = true;
                config.repeats = 5;
                config.lookups = 20_000;
                config.batch_size = 8_192;
            }
            "--repeats" => config.repeats = parse_value(&mut args, "--repeats"),
            "--lookups" => config.lookups = parse_value(&mut args, "--lookups"),
            "--batch-size" => config.batch_size = parse_value(&mut args, "--batch-size"),
            other => panic!("unknown argument: {other}"),
        }
    }
    assert!(config.repeats > 0, "--repeats must be greater than zero");
    assert!(config.lookups > 0, "--lookups must be greater than zero");
    assert!(
        config.batch_size > 0,
        "--batch-size must be greater than zero"
    );
    config
}

fn parse_value(args: &mut impl Iterator<Item = String>, name: &str) -> usize {
    args.next()
        .unwrap_or_else(|| panic!("{name} requires a value"))
        .parse()
        .unwrap_or_else(|_| panic!("{name} requires a positive integer"))
}

fn run_case(case: Case, config: &Config) {
    let (details, baseline_frags) = generate_details(case);
    let queries = sample_queries(&baseline_frags, config.lookups);
    let batch_source = queries
        .iter()
        .take(config.batch_size)
        .copied()
        .map(Some)
        .collect::<Vec<_>>();

    let legacy = LegacyFragReuseIndex::open(&details);
    let compact =
        CompactFragReuseIndex::try_new(Uuid::nil(), details.clone()).expect("valid benchmark FRI");
    for row_id in &queries {
        assert_eq!(
            legacy.remap_row_id(*row_id),
            compact.remap_row_id(*row_id),
            "legacy and compact semantics differ for case {} at row address {}",
            case.name,
            row_id
        );
    }

    emit_memory(case, "legacy_hash_map", legacy.deep_size_of());
    emit_memory(case, "compact_rank", compact.deep_size_of());

    // Warm allocator and code paths before collecting samples.
    black_box(LegacyFragReuseIndex::open(&details));
    black_box(CompactFragReuseIndex::try_new(Uuid::nil(), details.clone()).unwrap());

    let legacy_open = measure(config.repeats, || {
        black_box(LegacyFragReuseIndex::open(black_box(&details)));
    });
    let compact_open = measure(config.repeats, || {
        black_box(CompactFragReuseIndex::try_new(Uuid::nil(), black_box(details.clone())).unwrap());
    });
    emit_timing(case, "open_ns", "legacy_hash_map", 1, legacy_open);
    emit_timing(case, "open_ns", "compact_rank", 1, compact_open);

    let legacy_lookup = measure(config.repeats, || {
        for row_id in &queries {
            black_box(legacy.remap_row_id(black_box(*row_id)));
        }
    });
    let compact_lookup = measure(config.repeats, || {
        for row_id in &queries {
            black_box(compact.remap_row_id(black_box(*row_id)));
        }
    });
    emit_timing(
        case,
        "single_lookup_ns_per_row",
        "legacy_hash_map",
        queries.len(),
        legacy_lookup,
    );
    emit_timing(
        case,
        "single_lookup_ns_per_row",
        "compact_rank",
        queries.len(),
        compact_lookup,
    );

    let legacy_batch = measure(config.repeats, || {
        let mut batch = batch_source.clone();
        legacy.remap_row_ids_in_place(black_box(&mut batch));
        black_box(batch);
    });
    let compact_batch = measure(config.repeats, || {
        let mut batch = batch_source.clone();
        compact.remap_row_ids_in_place(black_box(&mut batch));
        black_box(batch);
    });
    emit_timing(
        case,
        "batch_remap_ns_per_row",
        "legacy_hash_map",
        batch_source.len(),
        legacy_batch,
    );
    emit_timing(
        case,
        "batch_remap_ns_per_row",
        "compact_rank",
        batch_source.len(),
        compact_batch,
    );
}

fn measure(mut repeats: usize, mut operation: impl FnMut()) -> Vec<u128> {
    let mut samples = Vec::with_capacity(repeats);
    while repeats > 0 {
        let start = Instant::now();
        operation();
        samples.push(start.elapsed().as_nanos());
        repeats -= 1;
    }
    samples
}

fn emit_memory(case: Case, implementation: &str, bytes: usize) {
    println!(
        "{}",
        json!({
            "type": "memory",
            "case": case.name,
            "rows": case.rows,
            "changed_basis_points": case.changed_basis_points,
            "chain_len": case.chain_len,
            "old_fragment_count": case.old_fragment_count,
            "groups_per_version": case.groups_per_version,
            "implementation": implementation,
            "retained_bytes_proxy": bytes,
        })
    );
}

fn emit_timing(
    case: Case,
    metric: &str,
    implementation: &str,
    operations: usize,
    samples_ns: Vec<u128>,
) {
    let mut normalized = samples_ns
        .iter()
        .map(|sample| *sample as f64 / operations as f64)
        .collect::<Vec<_>>();
    normalized.sort_by(f64::total_cmp);
    println!(
        "{}",
        json!({
            "type": "timing",
            "case": case.name,
            "rows": case.rows,
            "changed_basis_points": case.changed_basis_points,
            "chain_len": case.chain_len,
            "old_fragment_count": case.old_fragment_count,
            "groups_per_version": case.groups_per_version,
            "implementation": implementation,
            "metric": metric,
            "operations_per_sample": operations,
            "repeats": samples_ns.len(),
            "p50": percentile(&normalized, 0.50),
            "p99": percentile(&normalized, 0.99),
            "raw_total_ns": samples_ns,
        })
    );
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() as f64 * percentile).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn generate_details(case: Case) -> (FragReuseIndexDetails, Vec<FragDigest>) {
    let mut next_fragment_id = 1u64;
    let mut old_frags = distribute_rows(case.rows, case.old_fragment_count, &mut next_fragment_id);
    let baseline_frags = old_frags.clone();
    let mut versions = Vec::with_capacity(case.chain_len);

    for version_idx in 0..case.chain_len {
        let mut groups = Vec::new();
        let mut next_old_frags = Vec::new();
        let mut ordinal = 0u64;
        let group_size = old_frags.len().div_ceil(case.groups_per_version).max(1);
        for old_group in old_frags.chunks(group_size) {
            let mut changed = RoaringTreemap::new();
            let mut old_with_deletions = Vec::with_capacity(old_group.len());
            for frag in old_group {
                let mut num_changed = 0usize;
                for offset in 0..frag.physical_rows as u32 {
                    let hash = ordinal
                        .wrapping_mul(6_364_136_223_846_793_005)
                        .wrapping_add((version_idx as u64 + 1) * 1_442_695_040_888_963_407);
                    if (hash >> 32) % 10_000 < case.changed_basis_points as u64 {
                        changed.insert(u64::from(RowAddress::new_from_parts(
                            frag.id as u32,
                            offset,
                        )));
                        num_changed += 1;
                    }
                    ordinal += 1;
                }
                old_with_deletions.push(FragDigest {
                    id: frag.id,
                    physical_rows: frag.physical_rows,
                    num_deleted_rows: frag.physical_rows - num_changed,
                });
            }

            let num_changed = changed.len() as usize;
            let new_fragment_count = if num_changed == 0 {
                0
            } else {
                (old_group.len() + old_group.len() / 2)
                    .max(1)
                    .min(num_changed)
            };
            let new_frags = distribute_rows(num_changed, new_fragment_count, &mut next_fragment_id);
            let mut changed_row_addrs = Vec::with_capacity(changed.serialized_size());
            changed.serialize_into(&mut changed_row_addrs).unwrap();
            groups.push(FragReuseGroup {
                changed_row_addrs,
                old_frags: old_with_deletions,
                new_frags: new_frags.clone(),
            });
            next_old_frags.extend(new_frags);
        }

        versions.push(FragReuseVersion {
            dataset_version: version_idx as u64 + 1,
            groups,
        });
        old_frags = next_old_frags;
    }

    (FragReuseIndexDetails { versions }, baseline_frags)
}

fn distribute_rows(total_rows: usize, count: usize, next_id: &mut u64) -> Vec<FragDigest> {
    if count == 0 {
        return Vec::new();
    }
    let base = total_rows / count;
    let remainder = total_rows % count;
    (0..count)
        .map(|index| {
            let digest = FragDigest {
                id: *next_id,
                physical_rows: base + usize::from(index < remainder),
                num_deleted_rows: 0,
            };
            *next_id += 1;
            digest
        })
        .collect()
}

fn sample_queries(fragments: &[FragDigest], count: usize) -> Vec<u64> {
    let total_rows = fragments
        .iter()
        .map(|frag| frag.physical_rows)
        .sum::<usize>();
    let unaffected_fragment = u32::MAX - 1;
    let mut state = 0x4d595df4d0f33173u64;
    (0..count)
        .map(|index| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if index % 10 == 0 {
                return u64::from(RowAddress::new_from_parts(
                    unaffected_fragment,
                    state as u32,
                ));
            }
            let mut logical_row = state as usize % total_rows;
            for frag in fragments {
                if logical_row < frag.physical_rows {
                    return u64::from(RowAddress::new_from_parts(
                        frag.id as u32,
                        logical_row as u32,
                    ));
                }
                logical_row -= frag.physical_rows;
            }
            unreachable!("logical row is bounded by total_rows")
        })
        .collect()
}
