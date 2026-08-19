// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Tests for the logical scan planner.
//!
//! Most of these are equivalence tests: they build the same query through both the imperative and
//! the logical path and compare the rows. See [`harness`] for that oracle.

mod harness;
mod scan;
