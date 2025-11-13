// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use crate::dataset::refs::{normalize_branch, BranchContents};
use lance_core::Error;
use lance_core::Result;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Branch lineage tree representing parent → children relationships built from BranchContents snapshots.
///
/// Semantics:
/// - `branch`: node name (None denotes the virtual main root).
/// - `parent_version_number`: the version of the parent at which this child branch was created.
/// - `deleted`: indicates the node was inferred from a preserved snapshot (parent deleted in contents),
///   still kept to preserve lineage connectivity for inspection.
/// - `children`: ordered set for deterministic traversal.
#[derive(Debug)]
pub struct BranchLineage {
    pub deleted: bool,
    pub branch: Option<String>,
    pub parent_version_number: Option<u64>,
    pub children: BTreeSet<BranchLineage>,
}

impl BranchLineage {
    /// Postorder traversal (children before parent). Useful when cleaning from leaves upward
    /// or when summarizing child state before visiting the parent.
    pub fn post_order_iter(&self) -> PostOrderIter<'_> {
        PostOrderIter::new(self)
    }

    /// Preorder traversal (parent before children). This matches common inspection flows and cleanup
    /// reference scans: start at a root, then walk down descendants deterministically.
    ///
    /// Note: classic inorder (left → parent → right) doesn't make sense to branch lineage.
    pub fn pre_order_iter(&self) -> PreOrderIter<'_> {
        PreOrderIter::new(self)
    }

    pub fn post_order_iter_from(&self, root: Option<&str>) -> Result<PostOrderIter<'_>> {
        for node in self.pre_order_iter() {
            if node.branch.as_deref() == root {
                return Ok(PostOrderIter::new(node));
            }
        }
        Err(Error::RefNotFound {
            message: format!("Branch {} does not exist", normalize_branch(root)),
        })
    }

    pub fn pre_order_iter_from(&self, root: Option<&str>) -> Result<PreOrderIter<'_>> {
        for node in self.pre_order_iter() {
            if node.branch.as_deref() == root {
                return Ok(PreOrderIter::new(node));
            }
        }
        Err(Error::RefNotFound {
            message: format!("Branch {} does not exist", normalize_branch(root)),
        })
    }
}

pub struct PostOrderIter<'a> {
    stack: Vec<(&'a BranchLineage, usize)>,
}

impl<'a> PostOrderIter<'a> {
    pub fn new(root: &'a BranchLineage) -> Self {
        let mut iter = PostOrderIter { stack: Vec::new() };
        iter.push_with_index(root, 0);
        iter
    }

    fn push_with_index(&mut self, node: &'a BranchLineage, index: usize) {
        self.stack.push((node, index));
    }
}

impl<'a> Iterator for PostOrderIter<'a> {
    type Item = &'a BranchLineage;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some((node, index)) = self.stack.pop() {
            let children: Vec<&BranchLineage> = node.children.iter().collect();

            if index >= children.len() {
                return Some(node);
            } else {
                self.stack.push((node, index + 1));
                let child = children[index];
                self.stack.push((child, 0));
            }
        }
        None
    }
}

pub struct PreOrderIter<'a> {
    stack: Vec<&'a BranchLineage>,
}

impl<'a> PreOrderIter<'a> {
    pub fn new(root: &'a BranchLineage) -> Self {
        PreOrderIter { stack: vec![root] }
    }
}

impl<'a> Iterator for PreOrderIter<'a> {
    type Item = &'a BranchLineage;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.stack.pop()?;

        let children: Vec<&BranchLineage> = node.children.iter().collect();
        for child in children.into_iter().rev() {
            self.stack.push(child);
        }

        Some(node)
    }
}

impl PartialEq for BranchLineage {
    fn eq(&self, other: &Self) -> bool {
        self.parent_version_number == other.parent_version_number && self.branch == other.branch
    }
}

impl Eq for BranchLineage {}

impl PartialOrd for BranchLineage {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BranchLineage {
    fn cmp(&self, other: &Self) -> Ordering {
        let version_cmp = self.parent_version_number.cmp(&other.parent_version_number);
        if version_cmp != Ordering::Equal {
            return version_cmp;
        }
        self.branch.cmp(&other.branch)
    }
}

/// Build a BranchLineage tree from a full `BranchContents` map.
///
/// Principle:
/// - Invert `BranchContents.parent_branch` to create a parent → children mapping.
/// - Use `BranchContents.parent_lineage` snapshots to stitch in missing parents (marking them as `deleted = true`)
///   so the lineage remains connected even if some `BranchContents` files are removed.
/// - The virtual root (None) represents main.
///
/// Output:
/// - A deterministic tree capturing branch names, parent creation versions (`parent_version_number`),
///   and whether a node was inferred from a preserved snapshot (`deleted`).
pub fn collect_lineage_from(
    all_branch_contents: &HashMap<String, BranchContents>,
) -> Result<BranchLineage> {
    let mut parent_to_children: HashMap<Option<String>, HashSet<(&str, u64, bool)>> =
        HashMap::new();
    for (name, contents) in all_branch_contents {
        parent_to_children
            .entry(contents.parent_branch.clone())
            .or_default()
            .insert((name, contents.parent_version, false));
    }
    // In case some branches had been deleted.
    for contents in all_branch_contents.values() {
        let mut current_contents = contents;
        while let (Some(parent_branch_name), Some(parent_branch_contents)) = (
            current_contents.parent_branch.as_deref(),
            &current_contents.parent_lineage,
        ) {
            current_contents = parent_branch_contents;
            if all_branch_contents.contains_key(parent_branch_name) {
                continue;
            }

            parent_to_children
                .entry(parent_branch_contents.parent_branch.clone())
                .or_default()
                .insert((parent_branch_name, current_contents.parent_version, true));
        }
    }

    let mut main_lineage = BranchLineage {
        deleted: false,
        branch: None,
        parent_version_number: None,
        children: BTreeSet::new(),
    };

    fn expand(
        current: &mut BranchLineage,
        parent_to_children: &HashMap<Option<String>, HashSet<(&str, u64, bool)>>,
    ) {
        let parent_branch = &current.branch;
        if let Some(children) = parent_to_children.get(parent_branch) {
            for (child_name, parent_version, deleted) in children {
                let mut child_node = BranchLineage {
                    deleted: *deleted,
                    branch: Some(child_name.to_string()),
                    parent_version_number: Some(*parent_version),
                    children: BTreeSet::new(),
                };
                expand(&mut child_node, parent_to_children);
                current.children.insert(child_node);
            }
        }
    }

    expand(&mut main_lineage, &parent_to_children);
    Ok(main_lineage)
}

mod tests {
    use crate::dataset::branch_lineage::collect_lineage_from;
    use crate::dataset::refs::BranchContents;
    use std::collections::HashMap;

    /// Build a reusable mocked BranchContents map mirroring cleanup::lineage_tests::build_lineage_datasets.
    ///
    /// Structure:
    /// - main (virtual root)
    /// - branch1 -> main (parent_version = 1)
    /// - dev/branch2 -> branch1 (parent_version = 2)
    /// - feature/nathan/branch3 -> dev/branch2 (parent_version = 3)
    /// - branch4 -> main (parent_version = 2)
    ///
    /// Notes:
    /// - The "main" root is virtual (no BranchContents entry).
    /// - Version numbers are representative and monotonically increasing along the chain.
    /// - Tests reuse this builder to ensure consistent lineage and deterministic assertions.
    pub fn build_mock_branch_contents() -> HashMap<String, BranchContents> {
        fn build(
            parent_name: Option<&str>,
            parent_lineage: Option<&BranchContents>,
            parent_ver: u64,
        ) -> BranchContents {
            let parent_lineage = parent_lineage.map(|lineage| Box::new(lineage.clone()));
            BranchContents {
                parent_branch: parent_name.map(String::from),
                parent_lineage,
                parent_version: parent_ver,
                create_at: 0,
                manifest_size: 1,
            }
        }
        let mut contents = HashMap::new();
        contents.insert("branch1".to_string(), build(None, None, 1));
        contents.insert(
            "dev/branch2".to_string(),
            build(Some("branch1"), contents.get("branch1"), 2),
        );
        contents.insert(
            "feature/nathan/branch3".to_string(),
            build(Some("dev/branch2"), contents.get("dev/branch2"), 3),
        );
        contents.insert("branch4".to_string(), build(None, None, 2));
        contents
    }

    #[tokio::test]
    #[allow(clippy::bool_assert_comparison)]
    async fn test_collect_lineage_root_main() {
        let all_branches = build_mock_branch_contents();
        let lineage = collect_lineage_from(&all_branches).unwrap();

        let branch_lineages = lineage.post_order_iter().collect::<Vec<_>>();
        assert_eq!(branch_lineages.len(), 5);
        assert_eq!(
            branch_lineages[0].branch.as_deref(),
            Some("feature/nathan/branch3")
        );
        assert_eq!(branch_lineages[0].parent_version_number, Some(3));
        assert_eq!(branch_lineages[0].children.len(), 0);
        assert_eq!(branch_lineages[0].deleted, false);
        assert_eq!(branch_lineages[1].branch.as_deref(), Some("dev/branch2"));
        assert_eq!(branch_lineages[1].parent_version_number, Some(2));
        assert_eq!(branch_lineages[1].children.len(), 1);
        assert_eq!(branch_lineages[1].deleted, false);
        assert_eq!(branch_lineages[2].branch.as_deref(), Some("branch1"));
        assert_eq!(branch_lineages[2].parent_version_number, Some(1));
        assert_eq!(branch_lineages[2].children.len(), 1);
        assert_eq!(branch_lineages[2].deleted, false);
        assert_eq!(branch_lineages[3].branch.as_deref(), Some("branch4"));
        assert_eq!(branch_lineages[3].parent_version_number, Some(2));
        assert_eq!(branch_lineages[3].children.len(), 0);
        assert_eq!(branch_lineages[3].deleted, false);
        assert_eq!(branch_lineages[4].branch.as_deref(), None);
        assert_eq!(branch_lineages[4].parent_version_number, None);
        assert_eq!(branch_lineages[4].children.len(), 2);
        assert_eq!(branch_lineages[4].deleted, false);
    }

    #[tokio::test]
    #[allow(clippy::bool_assert_comparison)]
    async fn test_collect_lineage_root_branch3() {
        let all_branches = build_mock_branch_contents();
        let lineage = collect_lineage_from(&all_branches).unwrap();

        let branch_lineages = lineage
            .post_order_iter_from(Some("feature/nathan/branch3"))
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(branch_lineages.len(), 1);
        assert_eq!(
            branch_lineages[0].branch.as_deref(),
            Some("feature/nathan/branch3")
        );
        assert_eq!(branch_lineages[0].parent_version_number, Some(3));
        assert_eq!(branch_lineages[0].children.len(), 0);
        assert_eq!(branch_lineages[0].deleted, false);
    }

    #[tokio::test]
    #[allow(clippy::bool_assert_comparison)]
    async fn test_collect_lineage_root_branch2() {
        let all_branches = build_mock_branch_contents();
        let lineage = collect_lineage_from(&all_branches).unwrap();

        let branch_lineages = lineage
            .post_order_iter_from(Some("dev/branch2"))
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(branch_lineages.len(), 2);
        assert_eq!(
            branch_lineages[0].branch.as_deref(),
            Some("feature/nathan/branch3")
        );
        assert_eq!(branch_lineages[0].parent_version_number, Some(3));
        assert_eq!(branch_lineages[0].children.len(), 0);
        assert_eq!(branch_lineages[0].deleted, false);
        assert_eq!(branch_lineages[1].branch.as_deref(), Some("dev/branch2"));
        assert_eq!(branch_lineages[1].parent_version_number, Some(2));
        assert_eq!(branch_lineages[1].children.len(), 1);
        assert_eq!(branch_lineages[1].deleted, false);
    }

    #[tokio::test]
    #[allow(clippy::bool_assert_comparison)]
    async fn test_collect_lineage_root_branch1() {
        let all_branches = build_mock_branch_contents();
        let lineage = collect_lineage_from(&all_branches).unwrap();

        let branch_lineages = lineage
            .post_order_iter_from(Some("branch1"))
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(branch_lineages.len(), 3);
        assert_eq!(
            branch_lineages[0].branch.as_deref(),
            Some("feature/nathan/branch3")
        );
        assert_eq!(branch_lineages[0].parent_version_number, Some(3));
        assert_eq!(branch_lineages[0].children.len(), 0);
        assert_eq!(branch_lineages[0].deleted, false);
        assert_eq!(branch_lineages[1].branch.as_deref(), Some("dev/branch2"));
        assert_eq!(branch_lineages[1].parent_version_number, Some(2));
        assert_eq!(branch_lineages[1].children.len(), 1);
        assert_eq!(branch_lineages[1].deleted, false);
        assert_eq!(branch_lineages[2].branch.as_deref(), Some("branch1"));
        assert_eq!(branch_lineages[2].parent_version_number, Some(1));
        assert_eq!(branch_lineages[2].children.len(), 1);
        assert_eq!(branch_lineages[2].deleted, false);
    }

    #[tokio::test]
    #[allow(clippy::bool_assert_comparison)]
    async fn test_collect_lineage_root_branch4() {
        let all_branches = build_mock_branch_contents();
        let lineage = collect_lineage_from(&all_branches).unwrap();

        let branch_lineages = lineage
            .post_order_iter_from(Some("branch4"))
            .unwrap()
            .collect::<Vec<_>>();
        assert_eq!(branch_lineages.len(), 1);
        assert_eq!(branch_lineages[0].branch.as_deref(), Some("branch4"));
        assert_eq!(branch_lineages[0].parent_version_number, Some(2));
        assert_eq!(branch_lineages[0].children.len(), 0);
        assert_eq!(branch_lineages[0].deleted, false);
    }

    #[tokio::test]
    #[allow(clippy::bool_assert_comparison)]
    async fn test_collect_lineage_delete_branch1() {
        let mut all_branches = build_mock_branch_contents();
        all_branches.remove("branch1");
        let lineage = collect_lineage_from(&all_branches).unwrap();

        let branch_lineages = lineage.post_order_iter().collect::<Vec<_>>();
        assert_eq!(branch_lineages.len(), 5);
        assert_eq!(
            branch_lineages[0].branch.as_deref(),
            Some("feature/nathan/branch3")
        );
        assert_eq!(branch_lineages[0].parent_version_number, Some(3));
        assert_eq!(branch_lineages[0].children.len(), 0);
        assert_eq!(branch_lineages[0].deleted, false);
        assert_eq!(branch_lineages[1].branch.as_deref(), Some("dev/branch2"));
        assert_eq!(branch_lineages[1].parent_version_number, Some(2));
        assert_eq!(branch_lineages[1].children.len(), 1);
        assert_eq!(branch_lineages[1].deleted, false);
        assert_eq!(branch_lineages[2].branch.as_deref(), Some("branch1"));
        assert_eq!(branch_lineages[2].parent_version_number, Some(1));
        assert_eq!(branch_lineages[2].children.len(), 1);
        assert_eq!(branch_lineages[2].deleted, true);
        assert_eq!(branch_lineages[3].branch.as_deref(), Some("branch4"));
        assert_eq!(branch_lineages[3].parent_version_number, Some(2));
        assert_eq!(branch_lineages[3].children.len(), 0);
        assert_eq!(branch_lineages[3].deleted, false);
        assert_eq!(branch_lineages[4].branch.as_deref(), None);
        assert_eq!(branch_lineages[4].parent_version_number, None);
        assert_eq!(branch_lineages[4].children.len(), 2);
        assert_eq!(branch_lineages[4].deleted, false);
    }

    #[tokio::test]
    #[allow(clippy::bool_assert_comparison)]
    async fn test_collect_lineage_delete_branch1_and_branch3() {
        let mut all_branches = build_mock_branch_contents();
        all_branches.remove("branch1");
        all_branches.remove("feature/nathan/branch3");
        let lineage = collect_lineage_from(&all_branches).unwrap();

        let branch_lineages = lineage.post_order_iter().collect::<Vec<_>>();

        // Branch3 has no children, the lineage was deleted as well.
        assert_eq!(branch_lineages.len(), 4);
        assert_eq!(branch_lineages[0].branch.as_deref(), Some("dev/branch2"));
        assert_eq!(branch_lineages[0].parent_version_number, Some(2));
        assert_eq!(branch_lineages[0].children.len(), 0);
        assert_eq!(branch_lineages[0].deleted, false);
        assert_eq!(branch_lineages[1].branch.as_deref(), Some("branch1"));
        assert_eq!(branch_lineages[1].parent_version_number, Some(1));
        assert_eq!(branch_lineages[1].children.len(), 1);
        assert_eq!(branch_lineages[1].deleted, true);
        assert_eq!(branch_lineages[2].branch.as_deref(), Some("branch4"));
        assert_eq!(branch_lineages[2].parent_version_number, Some(2));
        assert_eq!(branch_lineages[2].children.len(), 0);
        assert_eq!(branch_lineages[2].deleted, false);
        assert_eq!(branch_lineages[3].branch.as_deref(), None);
        assert_eq!(branch_lineages[3].parent_version_number, None);
        assert_eq!(branch_lineages[3].children.len(), 2);
        assert_eq!(branch_lineages[3].deleted, false);
    }

    #[tokio::test]
    #[allow(clippy::bool_assert_comparison)]
    async fn test_collect_lineage_delete_branch1_and_branch4() {
        let mut all_branches = build_mock_branch_contents();
        all_branches.remove("branch1");
        all_branches.remove("branch4");
        let lineage = collect_lineage_from(&all_branches).unwrap();

        let branch_lineages = lineage.post_order_iter().collect::<Vec<_>>();
        assert_eq!(branch_lineages.len(), 4);
        assert_eq!(
            branch_lineages[0].branch.as_deref(),
            Some("feature/nathan/branch3")
        );
        assert_eq!(branch_lineages[0].parent_version_number, Some(3));
        assert_eq!(branch_lineages[0].children.len(), 0);
        assert_eq!(branch_lineages[0].deleted, false);
        assert_eq!(branch_lineages[1].branch.as_deref(), Some("dev/branch2"));
        assert_eq!(branch_lineages[1].parent_version_number, Some(2));
        assert_eq!(branch_lineages[1].children.len(), 1);
        assert_eq!(branch_lineages[1].deleted, false);
        assert_eq!(branch_lineages[2].branch.as_deref(), Some("branch1"));
        assert_eq!(branch_lineages[2].parent_version_number, Some(1));
        assert_eq!(branch_lineages[2].children.len(), 1);
        assert_eq!(branch_lineages[2].deleted, true);
        assert_eq!(branch_lineages[3].branch.as_deref(), None);
        assert_eq!(branch_lineages[3].parent_version_number, None);
        assert_eq!(branch_lineages[3].children.len(), 1);
        assert_eq!(branch_lineages[3].deleted, false);
    }

    #[tokio::test]
    #[allow(clippy::bool_assert_comparison)]
    async fn test_collect_lineage_and_bfs_scan() {
        let all_branches = build_mock_branch_contents();
        let lineage = collect_lineage_from(&all_branches).unwrap();

        let branch_lineages = lineage.pre_order_iter().collect::<Vec<_>>();
        assert_eq!(branch_lineages.len(), 5);
        assert_eq!(branch_lineages[0].branch.as_deref(), None);
        assert_eq!(branch_lineages[0].parent_version_number, None);
        assert_eq!(branch_lineages[0].children.len(), 2);
        assert_eq!(branch_lineages[0].deleted, false);
        assert_eq!(branch_lineages[1].branch.as_deref(), Some("branch1"));
        assert_eq!(branch_lineages[1].parent_version_number, Some(1));
        assert_eq!(branch_lineages[1].children.len(), 1);
        assert_eq!(branch_lineages[1].deleted, false);
        assert_eq!(branch_lineages[2].branch.as_deref(), Some("dev/branch2"));
        assert_eq!(branch_lineages[2].parent_version_number, Some(2));
        assert_eq!(branch_lineages[2].children.len(), 1);
        assert_eq!(branch_lineages[2].deleted, false);
        assert_eq!(
            branch_lineages[3].branch.as_deref(),
            Some("feature/nathan/branch3")
        );
        assert_eq!(branch_lineages[3].parent_version_number, Some(3));
        assert_eq!(branch_lineages[3].children.len(), 0);
        assert_eq!(branch_lineages[3].deleted, false);
        assert_eq!(branch_lineages[4].branch.as_deref(), Some("branch4"));
        assert_eq!(branch_lineages[4].parent_version_number, Some(2));
        assert_eq!(branch_lineages[4].children.len(), 0);
        assert_eq!(branch_lineages[4].deleted, false);
    }
}
