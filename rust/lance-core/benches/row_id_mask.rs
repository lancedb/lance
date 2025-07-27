// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lance_core::utils::mask::{MultiRowIdTreeMap, RowIdTreeMap};
use rand::{thread_rng, Rng};

fn generate_random_row_id_tree_map(
    num_fragments: u32,
    rows_per_fragment: u32,
    ordered_rows: bool,
) -> RowIdTreeMap {
    let mut rng = thread_rng();
    let mut map = RowIdTreeMap::new();
    for fragment in 0..num_fragments {
        let mut rows = Vec::with_capacity(rows_per_fragment as usize);
        if ordered_rows {
            for i in 0..rows_per_fragment {
                let row_id = ((fragment as u64) << 32) | (i as u64);
                rows.push(row_id);
            }
        } else {
            for _ in 0..rows_per_fragment {
                let row = rng.gen_range(0..u32::MAX);
                let row_id = ((fragment as u64) << 32) | (row as u64);
                rows.push(row_id);
            }
        }
        map.extend(rows);
    }
    map
}

fn bench_row_id_mask(c: &mut Criterion) {
    bench_union(c, false);
    bench_union(c, true);
}

fn bench_union(c: &mut Criterion, ordered_rows: bool) {
    let mut c = c.benchmark_group(format!("ordered_rows({})", ordered_rows));
    let num_maps = 10;
    let num_fragments = 5;
    let rows_per_fragment = 5000;
    let maps: Vec<RowIdTreeMap> = (0..num_maps)
        .map(|_| generate_random_row_id_tree_map(num_fragments, rows_per_fragment, ordered_rows))
        .collect();

    c.bench_function("RowIdTreeMap::union_all", |b| {
        b.iter(|| {
            black_box(RowIdTreeMap::union_all(maps.iter()));
        })
    });

    let unioned = RowIdTreeMap::union_all(maps.iter());
    let mut rng = thread_rng();
    c.bench_function("RowIdTreeMap::contains", |b| {
        b.iter(|| {
            black_box(unioned.contains(rng.gen()));
        })
    });

    let multi = MultiRowIdTreeMap::from_iter(maps.iter().map(|map| map.clone()));
    c.bench_function("MultiRowIdTreeMap::contains", |b| {
        b.iter(|| {
            black_box(multi.contains(rng.gen()));
        })
    });
}

criterion_group!(benches, bench_row_id_mask);
criterion_main!(benches);
