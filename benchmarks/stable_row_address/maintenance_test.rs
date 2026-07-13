// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

mod maintenance;

use std::num::NonZero;

use lance_table::format::{DataFile, Fragment};

fn fragment(id: u64, base_id: Option<u32>) -> Fragment {
    let mut fragment = Fragment::new(id);
    fragment.files.push(DataFile::new(
        format!("{id}.lance"),
        vec![0],
        vec![0],
        2,
        0,
        NonZero::new(1),
        base_id,
    ));
    fragment.physical_rows = Some(1);
    fragment
}

#[test]
fn no_stable_replay_uses_physical_fragment_order_within_policy_group() {
    let mut fragments = (0..8)
        .map(|id| fragment(id, Some(1)))
        .chain(std::iter::once(fragment(8, None)))
        .collect::<Vec<_>>();
    fragments.sort_by(maintenance::policy_fragment_order);

    assert_eq!(
        fragments
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        vec![8, 0, 1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        maintenance::fragments_for_native_compaction(&fragments, true)
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        (0..9).collect::<Vec<_>>()
    );
    assert_eq!(
        maintenance::fragments_for_native_compaction(&fragments, false)
            .iter()
            .map(|fragment| fragment.id)
            .collect::<Vec<_>>(),
        vec![8, 0, 1, 2, 3, 4, 5, 6, 7]
    );
}
