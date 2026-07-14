// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::time::Duration;

/// An operation result whose measured duration has already been frozen.
///
/// Post-operation evidence collection receives the result only through
/// [`Self::finish_with`], after `duration_ns` has been captured.
pub struct MeasuredOperation<T> {
    result: T,
    duration_ns: u64,
}

impl<T> MeasuredOperation<T> {
    pub fn freeze(result: T, elapsed: Duration) -> Self {
        Self {
            result,
            duration_ns: u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX),
        }
    }

    pub fn finish_with<U>(self, collect_evidence: impl FnOnce(T) -> U) -> (u64, U) {
        let duration_ns = self.duration_ns;
        let evidence = collect_evidence(self.result);
        (duration_ns, evidence)
    }
}
