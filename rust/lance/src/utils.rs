// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Various utilities

pub(crate) mod future;
pub(crate) mod temporal;
#[cfg(test)]
pub(crate) mod test;
#[cfg(feature = "tfrecord")]
pub mod tfrecord;

// Re-export
pub use lance_datafusion::sql;
pub use lance_linalg::kmeans;

pub fn default_deadlock_prevention_timeout() -> Option<std::time::Duration> {
    if let Ok(user_provided) =
        std::env::var("LANCE_DEADLOCK_PREVENTION").map(|val| val.parse::<u64>().unwrap())
    {
        if user_provided == 0 {
            None
        } else {
            Some(std::time::Duration::from_secs(user_provided))
        }
    } else {
        // By default don't do deadlock prevention.  It's too easy for
        // users to consume data slowly and we don't want to scare them
        // with a frightening log message
        None
    }
}
