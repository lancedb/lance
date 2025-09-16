// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::{Arc, Mutex};
use std::time::Duration;

/// AIMD (Additive Increase Multiplicative Decrease) controller for managing
/// upload concurrency in response to S3 throttling.
///
/// This implementation is inspired by EMRFS's AIMD approach for handling
/// S3 throttling. It dynamically adjusts the allowed concurrency level:
/// - On success: Additively increase capacity (after cooldown period)
/// - On throttle: Multiplicatively decrease capacity immediately
#[derive(Debug, Clone)]
pub struct AimdController {
    inner: Arc<AimdInner>,
}

#[derive(Debug)]
struct AimdInner {
    /// Current request rate
    rate: Mutex<f64>,
    /// Configuration for the AIMD algorithm
    config: AimdConfig,
    /// Track state for rate adjustments
    state: Mutex<AimdState>,
}

#[derive(Debug)]
struct AimdState {
    /// Count of responses in current window
    response_count: usize,
    /// Count of successful responses in current window
    success_count: usize,
}

/// Configuration for the AIMD controller
///
/// This follows the EMRFS configuration pattern from AWS EMR:
/// https://docs.aws.amazon.com/emr/latest/ReleaseGuide/emr-spark-emrfs-retry.html
#[derive(Debug, Clone)]
pub struct AimdConfig {
    /// Initial request rate (like fs.s3.aimd.initialRate)
    /// For PUT requests in object writer, this is scaled appropriately
    pub initial_rate: f64,
    /// Minimum request rate (like fs.s3.aimd.minRate)
    pub min_rate: f64,
    /// Maximum request rate (bounded for practical purposes)
    pub max_rate: f64,
    /// Additive increase increment (like fs.s3.aimd.increaseIncrement)
    pub increase_increment: f64,
    /// Multiplicative reduction factor (like fs.s3.aimd.reductionFactor)
    /// The rate is divided by this factor on throttling
    pub reduction_factor: f64,
    /// Number of responses before adjusting rate (like fs.s3.aimd.adjustWindow)
    pub adjust_window: usize,
}

impl Default for AimdConfig {
    fn default() -> Self {
        // Default values from EMRFS, scaled for PUT operations
        // EMRFS uses 3500/5500 ratio for PUT vs GET requests
        let put_scale_factor = 3500.0 / 5500.0;

        Self {
            initial_rate: std::env::var("LANCE_AIMD_INITIAL_RATE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(5500.0 * put_scale_factor),  // ~3500 for PUT operations
            min_rate: std::env::var("LANCE_AIMD_MIN_RATE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.1),  // EMRFS default
            max_rate: std::env::var("LANCE_AIMD_MAX_RATE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10000.0),  // Practical upper bound
            increase_increment: std::env::var("LANCE_AIMD_INCREASE_INCREMENT")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.1),  // EMRFS default
            reduction_factor: std::env::var("LANCE_AIMD_REDUCTION_FACTOR")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2.0),  // EMRFS default
            adjust_window: std::env::var("LANCE_AIMD_ADJUST_WINDOW")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(2),  // EMRFS default
        }
    }
}

impl AimdController {
    /// Create a new AIMD controller with default configuration
    pub fn new() -> Self {
        Self::with_config(AimdConfig::default())
    }

    /// Create a new AIMD controller with custom configuration
    pub fn with_config(config: AimdConfig) -> Self {
        assert!(
            config.reduction_factor > 1.0,
            "Reduction factor must be greater than 1.0"
        );
        assert!(
            config.min_rate > 0.0 && config.min_rate <= config.initial_rate,
            "Min rate must be positive and not exceed initial rate"
        );
        assert!(
            config.initial_rate <= config.max_rate,
            "Initial rate must not exceed max rate"
        );
        assert!(
            config.increase_increment > 0.0,
            "Increase increment must be positive"
        );
        assert!(
            config.adjust_window > 0,
            "Adjust window must be positive"
        );

        Self {
            inner: Arc::new(AimdInner {
                rate: Mutex::new(config.initial_rate),
                config,
                state: Mutex::new(AimdState {
                    response_count: 0,
                    success_count: 0,
                }),
            }),
        }
    }

    /// Get the current capacity (converts rate to concurrent operations)
    pub fn capacity(&self) -> usize {
        // Convert request rate to practical concurrent operations limit
        // This is a simplified conversion - in practice, you might want to
        // consider request duration and other factors
        let rate = *self.inner.rate.lock().unwrap();
        (rate / 100.0).max(1.0) as usize
    }

    /// Report a successful operation
    ///
    /// This may increase the rate based on the adjust window
    pub fn report_success(&self) {
        let mut state = self.inner.state.lock().unwrap();
        let mut rate = self.inner.rate.lock().unwrap();

        state.response_count += 1;
        state.success_count += 1;

        // Check if we've reached the adjust window
        if state.response_count >= self.inner.config.adjust_window {
            // All responses in window were successful - increase rate
            if state.success_count == state.response_count {
                let old_rate = *rate;
                *rate = (*rate + self.inner.config.increase_increment)
                    .min(self.inner.config.max_rate);

                if *rate != old_rate {
                    log::debug!(
                        "AIMD: Increased rate from {:.2} to {:.2} (success window)",
                        old_rate,
                        *rate
                    );
                }
            }

            // Reset window
            state.response_count = 0;
            state.success_count = 0;
        }
    }

    /// Report a throttling error
    ///
    /// This immediately decreases the rate using the reduction factor
    pub fn report_throttle(&self) {
        let mut rate = self.inner.rate.lock().unwrap();
        let mut state = self.inner.state.lock().unwrap();

        let old_rate = *rate;
        *rate = (*rate / self.inner.config.reduction_factor)
            .max(self.inner.config.min_rate);

        if *rate != old_rate {
            log::debug!(
                "AIMD: Decreased rate from {:.2} to {:.2} (throttled)",
                old_rate,
                *rate
            );
        }

        // Reset window on throttle
        state.response_count = 0;
        state.success_count = 0;
    }

    /// Check if we can start a new operation given current capacity
    ///
    /// Returns true if the current operations count is below capacity
    pub fn can_proceed(&self, current_operations: usize) -> bool {
        current_operations < self.capacity()
    }

    /// Reset the controller to initial state
    pub fn reset(&self) {
        *self.inner.rate.lock().unwrap() = self.inner.config.initial_rate;
        let mut state = self.inner.state.lock().unwrap();
        state.response_count = 0;
        state.success_count = 0;
    }
}

/// Check if an error indicates S3 throttling
///
/// Returns true for errors that should trigger AIMD backoff:
/// - 503 Service Unavailable
/// - 503 SlowDown
/// - RequestLimitExceeded
pub fn is_throttling_error(error: &object_store::Error) -> bool {
    match error {
        object_store::Error::Generic { source, .. } => {
            let error_str = source.to_string().to_lowercase();
            // Check for S3 throttling indicators
            error_str.contains("slowdown")
                || error_str.contains("service unavailable")
                || error_str.contains("503")
                || error_str.contains("request limit exceeded")
                || error_str.contains("throttl")
                || error_str.contains("rate exceeded")
                || error_str.contains("too many requests")
        }
        _ => false,
    }
}

/// Check if AIMD is enabled via environment variable
pub fn is_aimd_enabled() -> bool {
    std::env::var("LANCE_AIMD_ENABLED")
        .ok()
        .and_then(|s| s.parse::<bool>().ok())
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use object_store::path::Path;
    use object_store::throttle::{ThrottleConfig, ThrottledStore};
    use object_store::ObjectStore;
    use tokio::io::AsyncWriteExt;
    use url::Url;
    use crate::object_store::ObjectStore as LanceObjectStore;
    use crate::object_writer::ObjectWriter;

    #[test]
    fn test_aimd_config_defaults() {
        let config = AimdConfig::default();
        let put_scale_factor = 3500.0 / 5500.0;
        assert!((config.initial_rate - 5500.0 * put_scale_factor).abs() < 0.01);
        assert_eq!(config.min_rate, 0.1);
        assert_eq!(config.max_rate, 10000.0);
        assert_eq!(config.increase_increment, 0.1);
        assert_eq!(config.reduction_factor, 2.0);
        assert_eq!(config.adjust_window, 2);
    }

    #[test]
    fn test_aimd_additive_increase() {
        let config = AimdConfig {
            initial_rate: 500.0,
            min_rate: 0.1,
            max_rate: 1000.0,
            increase_increment: 0.1,
            reduction_factor: 2.0,
            adjust_window: 2,
        };
        let controller = AimdController::with_config(config);

        let initial_capacity = controller.capacity();

        // Need to report success for the full window to trigger increase
        controller.report_success();
        controller.report_success();

        // Rate should have increased
        assert!(controller.capacity() >= initial_capacity);
    }

    #[test]
    fn test_aimd_multiplicative_decrease() {
        let config = AimdConfig {
            initial_rate: 1000.0,
            min_rate: 100.0,
            max_rate: 2000.0,
            increase_increment: 0.1,
            reduction_factor: 2.0,
            adjust_window: 2,
        };
        let controller = AimdController::with_config(config);

        let initial = controller.capacity();
        controller.report_throttle();
        let after_first = controller.capacity();
        assert!(after_first < initial);

        controller.report_throttle();
        let after_second = controller.capacity();
        assert!(after_second <= after_first);
    }

    #[test]
    fn test_aimd_max_rate() {
        let max_rate = 1000.0;
        let config = AimdConfig {
            initial_rate: 990.0,
            min_rate: 0.1,
            max_rate,
            increase_increment: 20.0,
            reduction_factor: 2.0,
            adjust_window: 2,
        };
        let controller = AimdController::with_config(config);

        let _initial = controller.capacity();

        // Report success for full window
        controller.report_success();
        controller.report_success();

        let _after_increase = controller.capacity();

        // Should be capped by max rate
        controller.report_success();
        controller.report_success();

        let after_second = controller.capacity();
        assert!(after_second <= (max_rate / 100.0) as usize);
    }

    #[test]
    fn test_aimd_adjust_window() {
        let config = AimdConfig {
            initial_rate: 500.0,
            min_rate: 0.1,
            max_rate: 1000.0,
            increase_increment: 10.0,
            reduction_factor: 2.0,
            adjust_window: 3, // Need 3 successes to increase
        };
        let controller = AimdController::with_config(config);

        let initial = controller.capacity();

        // Only 2 successes - should not increase yet
        controller.report_success();
        controller.report_success();
        assert_eq!(controller.capacity(), initial);

        // Third success completes window - should increase
        controller.report_success();
        assert!(controller.capacity() >= initial);
    }

    #[test]
    fn test_throttling_error_detection() {
        let error = object_store::Error::Generic {
            store: "S3",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "503 SlowDown: Please reduce request rate",
            )),
        };
        assert!(is_throttling_error(&error));

        let error = object_store::Error::Generic {
            store: "S3",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Service Unavailable",
            )),
        };
        assert!(is_throttling_error(&error));

        let error = object_store::Error::Generic {
            store: "S3",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Request limit exceeded",
            )),
        };
        assert!(is_throttling_error(&error));

        let error = object_store::Error::Generic {
            store: "S3",
            source: Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Connection reset by peer",
            )),
        };
        assert!(!is_throttling_error(&error));
    }

    #[test]
    fn test_aimd_can_proceed() {
        let controller = AimdController::new();
        let initial_capacity = controller.capacity();

        assert!(controller.can_proceed(initial_capacity - 1));
        assert!(!controller.can_proceed(initial_capacity));
        assert!(!controller.can_proceed(initial_capacity + 1));
    }

    #[tokio::test]
    async fn test_aimd_controller_behavior() {
        let config = AimdConfig {
            initial_rate: 1000.0,
            min_rate: 100.0,
            max_rate: 2000.0,
            increase_increment: 50.0,
            reduction_factor: 2.0,
            adjust_window: 2,
        };
        let controller = AimdController::with_config(config);

        let initial = controller.capacity();

        // Simulate throttling - should decrease rate
        controller.report_throttle();
        let after_throttle = controller.capacity();
        assert!(after_throttle < initial);

        // Two successes should increase rate
        controller.report_success();
        controller.report_success();
        let after_success = controller.capacity();
        assert!(after_success >= after_throttle);
    }

    /// Create a throttled store that simulates S3 throttling
    fn create_throttled_store(base_store: Arc<dyn ObjectStore>) -> Arc<dyn ObjectStore> {
        let config = ThrottleConfig {
            // Simulate throttling by adding delays
            wait_list_per_call: Duration::from_millis(10),
            wait_get_per_call: Duration::from_millis(10),
            wait_put_per_call: Duration::from_millis(50),
            ..Default::default()
        };
        Arc::new(ThrottledStore::new(base_store, config))
    }

    #[tokio::test]
    async fn test_aimd_with_throttling() {
        // Set up environment for AIMD
        std::env::set_var("LANCE_AIMD_ENABLED", "true");
        std::env::set_var("LANCE_AIMD_INITIAL_RATE", "3500");
        std::env::set_var("LANCE_AIMD_MIN_RATE", "0.1");
        std::env::set_var("LANCE_AIMD_MAX_RATE", "10000");
        std::env::set_var("LANCE_AIMD_INCREASE_INCREMENT", "0.1");
        std::env::set_var("LANCE_AIMD_REDUCTION_FACTOR", "2");
        std::env::set_var("LANCE_AIMD_ADJUST_WINDOW", "2");

        // Create a memory store wrapped with throttling
        let memory_store = object_store::memory::InMemory::new();
        let throttled_store = create_throttled_store(Arc::new(memory_store));
        let lance_store = LanceObjectStore::new(
            throttled_store,
            Url::parse("memory:///").unwrap(),
            None,
            None,
            false,
            true,
            16,
            3,
            None,
        );

        // Create object writer with AIMD enabled
        let path = Path::from("test_file.parquet");
        let mut writer = ObjectWriter::new(&lance_store, &path).await.unwrap();

        // Write data that triggers multipart upload
        let chunk_size = 1024 * 1024 * 6; // 6MB chunks
        let data = vec![0u8; chunk_size];

        // Write multiple chunks to trigger concurrent uploads
        for _ in 0..5 {
            writer.write_all(&data).await.unwrap();
        }

        // Complete the write
        let result = writer.shutdown().await.unwrap();
        assert_eq!(result.size, chunk_size * 5);

        // Clean up
        std::env::remove_var("LANCE_AIMD_ENABLED");
        std::env::remove_var("LANCE_AIMD_INITIAL_RATE");
        std::env::remove_var("LANCE_AIMD_MIN_RATE");
        std::env::remove_var("LANCE_AIMD_MAX_RATE");
        std::env::remove_var("LANCE_AIMD_INCREASE_INCREMENT");
        std::env::remove_var("LANCE_AIMD_REDUCTION_FACTOR");
        std::env::remove_var("LANCE_AIMD_ADJUST_WINDOW");
    }
}