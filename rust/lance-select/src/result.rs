// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Interval-shaped wrappers around a row-address mask returned by a
//! scalar-index expression evaluation.
//!
//! Each result describes a closed interval `[lower, upper]` in the
//! lattice of subsets:
//!
//! * `lower` — rows the index *guarantees* are in the answer.
//! * `upper` — rows that *might* be in the answer; rows outside `upper`
//!   are guaranteed not in the answer.
//!
//! The three pre-existing "shapes" map onto degenerate intervals:
//!
//! | Old variant | Interval form                          |
//! |-------------|----------------------------------------|
//! | `Exact(m)`  | `{lower: m, upper: m}`                 |
//! | `AtMost(m)` | `{lower: allow_nothing(), upper: m}`   |
//! | `AtLeast(m)`| `{lower: m, upper: all_rows()}`        |
//!
//! Use [`IndexExprResult::exact`] / [`IndexExprResult::at_most`] /
//! [`IndexExprResult::at_least`] to construct those shapes, and the
//! matching [`IndexExprResult::is_exact`] etc. predicates to inspect
//! them. Intervals that are neither (the "Refined" case — a non-empty
//! `lower` strictly inside a non-universe `upper`) arise from indices
//! that can distinguish guaranteed-match from candidate-match rows
//! within a single search (e.g. a zone map answering `IS NOT NULL`).
//!
//! The boolean algebra (`Not` / `BitAnd` / `BitOr`) is elementwise on
//! the endpoints:
//!
//! ```text
//! !{l, u}                = {!u, !l}
//! {l1, u1} & {l2, u2}    = {l1 & l2, u1 & u2}
//! {l1, u1} | {l2, u2}    = {l1 | l2, u1 | u2}
//! ```
//!
//! This works for both the post-`drop_nulls` form ([`IndexExprResult`],
//! backed by [`RowAddrMask`]) and the during-evaluation form
//! ([`NullableIndexExprResult`], backed by [`NullableRowAddrMask`]) —
//! the per-endpoint algebra already implements two-valued and SQL
//! three-valued logic correctly inside each mask type.

use crate::mask::{NullableRowAddrMask, RowAddrMask, RowSetOps};

/// Result of an index search before NULL rows are dropped. Each endpoint
/// is a [`NullableRowAddrMask`] carrying SQL three-valued logic info.
#[derive(Debug, Clone)]
pub struct NullableIndexExprResult {
    /// Rows the index *guarantees* are TRUE.
    pub lower: NullableRowAddrMask,
    /// Rows that may be TRUE. Rows outside `upper` are guaranteed to be
    /// FALSE / NULL (and so not in a `WHERE` answer set).
    pub upper: NullableRowAddrMask,
}

impl NullableIndexExprResult {
    /// Precise result — every row in `mask` is in the answer and every
    /// row outside is not. Equivalent to the old `Exact` variant.
    pub fn exact(mask: NullableRowAddrMask) -> Self {
        Self {
            lower: mask.clone(),
            upper: mask,
        }
    }

    /// Upper-bound-only result — rows outside `mask` are guaranteed not
    /// to match; rows inside may match and require a recheck.
    /// Equivalent to the old `AtMost` variant.
    pub fn at_most(mask: NullableRowAddrMask) -> Self {
        Self {
            lower: NullableRowAddrMask::allow_nothing(),
            upper: mask,
        }
    }

    /// Lower-bound-only result — rows in `mask` are guaranteed to match;
    /// rows outside may match too and require a recheck. Equivalent to
    /// the old `AtLeast` variant.
    pub fn at_least(mask: NullableRowAddrMask) -> Self {
        Self {
            lower: mask,
            upper: NullableRowAddrMask::all_rows(),
        }
    }

    /// True if `lower == upper` — the answer is precisely the lower
    /// (== upper) mask.
    ///
    /// This is a **structural** check on the canonical form produced by
    /// the constructors / algebra: an `Exact(m)` built with
    /// [`Self::exact`] holds equal masks, and elementwise `&` / `|` / `!`
    /// preserve that. It is not a semantic emptiness test — a
    /// hand-constructed `IndexExprResult` whose endpoints are
    /// representationally distinct but semantically equal (e.g.
    /// `AllowList(universe)` vs `BlockList(empty)`) will report
    /// `is_exact() == false`. All in-tree code paths construct results
    /// through the canonical builders, so this is sound in practice.
    ///
    /// The three shape predicates are not mutually exclusive — see the
    /// note on [`Self::is_at_least`] for the precedence convention.
    pub fn is_exact(&self) -> bool {
        self.lower == self.upper
    }

    /// True if `lower` matches no rows (canonical `AllowList(∅)`) — the
    /// index gives only an upper bound on the answer.
    ///
    /// Like [`Self::is_exact`], this is a structural check on the
    /// canonical form. See that doc for the caveat.
    pub fn is_at_most(&self) -> bool {
        matches!(&self.lower, NullableRowAddrMask::AllowList(set) if set.is_empty())
    }

    /// True if `upper` covers every row (canonical `BlockList(∅)`) — the
    /// index gives only a lower bound on the answer.
    ///
    /// **Precedence convention** for consumers branching on shape: check
    /// [`Self::is_exact`] *first* (Exact-of-empty satisfies both
    /// `is_exact` and `is_at_most`; Exact-of-universe satisfies both
    /// `is_exact` and `is_at_least`); then `is_at_least`; finally treat
    /// the residual as `is_at_most` or Refined. The branches in
    /// `filtered_read::apply_index_to_fragment` follow this order.
    pub fn is_at_least(&self) -> bool {
        matches!(&self.upper, NullableRowAddrMask::BlockList(set) if set.is_empty())
    }

    /// Project NULL rows out of the result.
    ///
    /// Under a `WHERE` clause NULL is treated as FALSE, so `drop_nulls`
    /// folds NULL rows out of the answer at each endpoint.
    pub fn drop_nulls(self) -> IndexExprResult {
        IndexExprResult {
            lower: self.lower.drop_nulls(),
            upper: self.upper.drop_nulls(),
        }
    }
}

impl std::ops::Not for NullableIndexExprResult {
    type Output = Self;

    fn not(self) -> Self {
        Self {
            lower: !self.upper,
            upper: !self.lower,
        }
    }
}

impl std::ops::BitAnd<Self> for NullableIndexExprResult {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self {
            lower: self.lower & rhs.lower,
            upper: self.upper & rhs.upper,
        }
    }
}

impl std::ops::BitOr<Self> for NullableIndexExprResult {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self {
            lower: self.lower | rhs.lower,
            upper: self.upper | rhs.upper,
        }
    }
}

/// Result of an index search after NULL rows have been dropped. This is
/// what the read planner consumes.
#[derive(Debug, Clone)]
pub struct IndexExprResult {
    /// Rows the index *guarantees* are in the answer.
    pub lower: RowAddrMask,
    /// Rows that may be in the answer. Rows outside `upper` are
    /// guaranteed not in the answer.
    pub upper: RowAddrMask,
}

impl IndexExprResult {
    /// Precise result — every row in `mask` is in the answer and every
    /// row outside is not. Equivalent to the old `Exact` variant.
    pub fn exact(mask: RowAddrMask) -> Self {
        Self {
            lower: mask.clone(),
            upper: mask,
        }
    }

    /// Upper-bound-only result. Equivalent to the old `AtMost` variant.
    pub fn at_most(mask: RowAddrMask) -> Self {
        Self {
            lower: RowAddrMask::allow_nothing(),
            upper: mask,
        }
    }

    /// Lower-bound-only result. Equivalent to the old `AtLeast` variant.
    pub fn at_least(mask: RowAddrMask) -> Self {
        Self {
            lower: mask,
            upper: RowAddrMask::all_rows(),
        }
    }

    /// True if `lower == upper` — the answer is precisely the lower
    /// (== upper) mask. See [`NullableIndexExprResult::is_exact`] for the
    /// structural-form caveat and the precedence convention shared with
    /// [`Self::is_at_most`] / [`Self::is_at_least`].
    pub fn is_exact(&self) -> bool {
        self.lower == self.upper
    }

    /// True if `lower` matches no rows (canonical `AllowList(∅)`) — the
    /// index gives only an upper bound on the answer. See
    /// [`NullableIndexExprResult::is_exact`] for caveats.
    pub fn is_at_most(&self) -> bool {
        matches!(&self.lower, RowAddrMask::AllowList(set) if set.is_empty())
    }

    /// True if `upper` covers every row (canonical `BlockList(∅)`) — the
    /// index gives only a lower bound on the answer. See
    /// [`NullableIndexExprResult::is_at_least`] for the precedence
    /// convention consumers should follow.
    pub fn is_at_least(&self) -> bool {
        matches!(&self.upper, RowAddrMask::BlockList(set) if set.is_empty())
    }
}

impl std::ops::Not for IndexExprResult {
    type Output = Self;

    fn not(self) -> Self {
        Self {
            lower: !self.upper,
            upper: !self.lower,
        }
    }
}

impl std::ops::BitAnd<Self> for IndexExprResult {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        Self {
            lower: self.lower & rhs.lower,
            upper: self.upper & rhs.upper,
        }
    }
}

impl std::ops::BitOr<Self> for IndexExprResult {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self {
            lower: self.lower | rhs.lower,
            upper: self.upper | rhs.upper,
        }
    }
}
