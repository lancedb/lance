// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Trait for systems that can display progress of long running operations.
///
/// The long running operation should take in Arc<dyn GenericProgressCallback>
/// and call the `begin` and `update` methods to report progress.
///
/// The `begin` method should be called once at the beginning of the operation.
pub trait GenericProgressCallback: Send + Sync {
    /// Called when the operation starts
    fn begin(&self, total_units: u64);
    /// Called when some work has been completed
    ///
    /// new_units_completed represents the number of units completed since the last update.
    fn update(&self, new_units_completed: u64);
}

#[derive(Default)]
pub struct NoopProgressCallback {}

impl GenericProgressCallback for NoopProgressCallback {
    fn begin(&self, _total_units: u64) {}

    fn update(&self, _new_units_completed: u64) {}
}
