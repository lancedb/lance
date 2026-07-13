// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::cmp::Ordering;

use lance_table::format::Fragment;

pub fn policy_fragment_order(left: &Fragment, right: &Fragment) -> Ordering {
    let left_file = left.files.first();
    let right_file = right.files.first();
    (
        left_file.and_then(|file| file.base_id),
        left.id,
        left_file.map(|file| file.path.as_str()).unwrap_or_default(),
    )
        .cmp(&(
            right_file.and_then(|file| file.base_id),
            right.id,
            right_file
                .map(|file| file.path.as_str())
                .unwrap_or_default(),
        ))
}

pub fn fragments_for_native_compaction(
    sources: &[Fragment],
    requires_physical_address_order: bool,
) -> Vec<Fragment> {
    let mut fragments = sources.to_vec();
    if requires_physical_address_order {
        fragments.sort_by_key(|fragment| fragment.id);
    }
    fragments
}
