// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Plan-time expression parsing for scalar-index pushdown.
//!
//! [`scalar`] splits a `WHERE` clause into an index-evaluable leaf plus a
//! refine residual. The aggregate-side pushdown work currently lives in
//! `lance::io::exec::count_pushdown` and is narrowly scoped to the
//! count-from-mask category; future categories (mask-to-answer,
//! zone-aware, dimension-keyed) would each grow their own machinery.

pub mod scalar;
