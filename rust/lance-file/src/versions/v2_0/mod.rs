// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance v2.0 encoding composition.

use std::sync::Arc;

use lance_encoding::{array_encoding::ArrayFieldEncodingStrategy, encoder::FieldEncodingStrategy};

/// Compose the v2.0 field encoding mechanisms.
pub fn encoding_strategy() -> Arc<dyn FieldEncodingStrategy> {
    Arc::new(ArrayFieldEncodingStrategy::new())
}
