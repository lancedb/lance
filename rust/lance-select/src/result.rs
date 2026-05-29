// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Certainty-tagged wrappers around a row-address mask returned by a
//! scalar-index expression evaluation.
//!
//! These types model the three possible degrees of knowledge an index
//! search can return:
//!
//! * [`Exact`] — the mask is the precise answer; no recheck needed.
//! * [`AtMost`] — the mask is a *superset* of the true answer; the rows
//!   inside the mask must be rechecked against the predicate.
//! * [`AtLeast`] — the mask is a *subset* of the true answer; the rows
//!   outside the mask must be rechecked against the predicate.
//!
//! The boolean algebra (`Not`/`BitAnd`/`BitOr`) is implemented on both
//! [`NullableIndexExprResult`] (the form during evaluation, carrying SQL
//! three-valued logic via [`NullableRowAddrMask`]) and
//! [`IndexExprResult`] (the form consumed by the read planner, after
//! `drop_nulls` collapses NULL rows into FALSE).
//!
//! [`Exact`]: IndexExprResult::Exact
//! [`AtMost`]: IndexExprResult::AtMost
//! [`AtLeast`]: IndexExprResult::AtLeast

use crate::mask::{NullableRowAddrMask, RowAddrMask};

/// Result of an index search before NULL rows are dropped. Carries
/// three-valued-logic information via [`NullableRowAddrMask`].
#[derive(Debug)]
pub enum NullableIndexExprResult {
    Exact(NullableRowAddrMask),
    AtMost(NullableRowAddrMask),
    AtLeast(NullableRowAddrMask),
}

impl std::ops::Not for NullableIndexExprResult {
    type Output = Self;

    fn not(self) -> Self {
        // Flip certainty: NOT(AtMost) → AtLeast, NOT(AtLeast) → AtMost.
        // NULL info is preserved by `NullableRowAddrMask::not` (it flips
        // AllowList ↔ BlockList without touching the `nulls` field), which
        // is the 3VL-correct negation: TRUE↔FALSE swap, NULL stays NULL.
        match self {
            Self::Exact(mask) => Self::Exact(!mask),
            Self::AtMost(mask) => Self::AtLeast(!mask),
            Self::AtLeast(mask) => Self::AtMost(!mask),
        }
    }
}

impl std::ops::BitAnd<Self> for NullableIndexExprResult {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Exact(lhs), Self::Exact(rhs)) => Self::Exact(lhs & rhs),
            (Self::Exact(lhs), Self::AtMost(rhs)) | (Self::AtMost(lhs), Self::Exact(rhs)) => {
                Self::AtMost(lhs & rhs)
            }
            (Self::Exact(exact), Self::AtLeast(_)) | (Self::AtLeast(_), Self::Exact(exact)) => {
                // We could do better here, elements in both lhs and rhs are known
                // to be true and don't require a recheck.  We only need to recheck
                // elements in lhs that are not in rhs
                Self::AtMost(exact)
            }
            (Self::AtMost(lhs), Self::AtMost(rhs)) => Self::AtMost(lhs & rhs),
            (Self::AtLeast(lhs), Self::AtLeast(rhs)) => Self::AtLeast(lhs & rhs),
            (Self::AtMost(most), Self::AtLeast(_)) | (Self::AtLeast(_), Self::AtMost(most)) => {
                Self::AtMost(most)
            }
        }
    }
}

impl std::ops::BitOr<Self> for NullableIndexExprResult {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Exact(lhs), Self::Exact(rhs)) => Self::Exact(lhs | rhs),
            (Self::Exact(lhs), Self::AtMost(rhs)) | (Self::AtMost(rhs), Self::Exact(lhs)) => {
                // We could do better here, elements in lhs are known to be true
                // and don't require a recheck.  We only need to recheck elements
                // in rhs that are not in lhs
                Self::AtMost(lhs | rhs)
            }
            (Self::Exact(lhs), Self::AtLeast(rhs)) | (Self::AtLeast(rhs), Self::Exact(lhs)) => {
                Self::AtLeast(lhs | rhs)
            }
            (Self::AtMost(lhs), Self::AtMost(rhs)) => Self::AtMost(lhs | rhs),
            (Self::AtLeast(lhs), Self::AtLeast(rhs)) => Self::AtLeast(lhs | rhs),
            (Self::AtMost(_), Self::AtLeast(least)) | (Self::AtLeast(least), Self::AtMost(_)) => {
                Self::AtLeast(least)
            }
        }
    }
}

impl NullableIndexExprResult {
    /// Project NULL rows out of the result.
    ///
    /// Under a `WHERE` clause, NULL is treated as FALSE — so `drop_nulls`
    /// removes them from `AllowList`s (NULL rows are not selected) and
    /// folds them into `BlockList`s (NULL rows are still blocked).
    pub fn drop_nulls(self) -> IndexExprResult {
        match self {
            Self::Exact(mask) => IndexExprResult::Exact(mask.drop_nulls()),
            Self::AtMost(mask) => IndexExprResult::AtMost(mask.drop_nulls()),
            Self::AtLeast(mask) => IndexExprResult::AtLeast(mask.drop_nulls()),
        }
    }
}

/// Result of an index search after NULL rows have been dropped. This is
/// what the read planner consumes.
#[derive(Debug)]
pub enum IndexExprResult {
    /// The answer is exactly the rows in the allow list minus the rows
    /// in the block list.
    Exact(RowAddrMask),
    /// The answer is at most the rows in the allow list minus the rows
    /// in the block list. Some of the rows in the allow list may not be
    /// in the result and will need to be filtered by a recheck. Every
    /// row in the block list is definitely not in the result.
    AtMost(RowAddrMask),
    /// The answer is at least the rows in the allow list minus the rows
    /// in the block list. Some of the rows in the block list might be in
    /// the result. Every row in the allow list is definitely in the
    /// result.
    AtLeast(RowAddrMask),
}

impl IndexExprResult {
    pub fn row_addr_mask(&self) -> &RowAddrMask {
        match self {
            Self::Exact(mask) => mask,
            Self::AtMost(mask) => mask,
            Self::AtLeast(mask) => mask,
        }
    }

    pub fn discriminant(&self) -> u32 {
        match self {
            Self::Exact(_) => 0,
            Self::AtMost(_) => 1,
            Self::AtLeast(_) => 2,
        }
    }
}
