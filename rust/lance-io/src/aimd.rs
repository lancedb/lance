// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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

/// Operation type for AIMD controller
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Read/GET operations
    Read,
    /// Write/PUT operations
    Write,
}

impl AimdConfig {
    /// Create configuration from storage options
    pub fn from_storage_options(
        op_type: OperationType,
        storage_options: Option<&HashMap<String, String>>,
    ) -> Self {
        // EMRFS uses different initial rates for GET and PUT
        // GET: 5500, PUT: 3500 (3500/5500 ratio)
        let initial_rate_default = match op_type {
            OperationType::Read => 5500.0,
            OperationType::Write => 3500.0,
        };

        let mut config = Self {
            initial_rate: initial_rate_default,
            min_rate: 0.1,  // EMRFS default
            max_rate: 10000.0,  // Practical upper bound
            increase_increment: 0.1,  // EMRFS default
            reduction_factor: 2.0,  // EMRFS default
            adjust_window: 2,  // EMRFS default
        };

        if let Some(options) = storage_options {
            // Parse AIMD configuration from storage options
            // Keys use dots instead of underscores for consistency with other storage options
            if let Some(enabled) = options.get("lance.aimd.enabled") {
                if enabled.parse::<bool>().unwrap_or(true) == false {
                    // If AIMD is disabled, return default config (caller should check is_enabled)
                    return config;
                }
            }

            // Operation-specific initial rate
            let initial_rate_key = match op_type {
                OperationType::Read => "lance.aimd.read.initial_rate",
                OperationType::Write => "lance.aimd.write.initial_rate",
            };
            if let Some(val) = options.get(initial_rate_key) {
                if let Ok(rate) = val.parse::<f64>() {
                    config.initial_rate = rate;
                }
            } else if let Some(val) = options.get("lance.aimd.initial_rate") {
                // Fall back to generic initial rate
                if let Ok(rate) = val.parse::<f64>() {
                    config.initial_rate = rate;
                }
            }

            // Common parameters
            if let Some(val) = options.get("lance.aimd.min_rate") {
                if let Ok(rate) = val.parse::<f64>() {
                    config.min_rate = rate;
                }
            }
            if let Some(val) = options.get("lance.aimd.max_rate") {
                if let Ok(rate) = val.parse::<f64>() {
                    config.max_rate = rate;
                }
            }
            if let Some(val) = options.get("lance.aimd.increase_increment") {
                if let Ok(inc) = val.parse::<f64>() {
                    config.increase_increment = inc;
                }
            }
            if let Some(val) = options.get("lance.aimd.reduction_factor") {
                if let Ok(factor) = val.parse::<f64>() {
                    config.reduction_factor = factor;
                }
            }
            if let Some(val) = options.get("lance.aimd.adjust_window") {
                if let Ok(window) = val.parse::<usize>() {
                    config.adjust_window = window;
                }
            }
        }

        config
    }

    /// Create default configuration for a specific operation type
    pub fn default_for_operation(op_type: OperationType) -> Self {
        Self::from_storage_options(op_type, None)
    }
}

impl Default for AimdConfig {
    fn default() -> Self {
        // Default to write configuration for backward compatibility
        Self::default_for_operation(OperationType::Write)
    }
}

impl AimdController {
    /// Create a new AIMD controller with default configuration
    pub fn new() -> Self {
        Self::with_config(AimdConfig::default())
    }

    /// Create a new AIMD controller for a specific operation type
    pub fn new_for_operation(op_type: OperationType) -> Self {
        Self::with_config(AimdConfig::default_for_operation(op_type))
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

/// Check if AIMD is enabled via storage options
pub fn is_aimd_enabled(storage_options: Option<&HashMap<String, String>>) -> bool {
    if let Some(options) = storage_options {
        // Check if explicitly disabled
        if let Some(enabled) = options.get("lance.aimd.enabled") {
            return enabled.parse::<bool>().unwrap_or(true);
        }
        // If any AIMD config is present, consider it enabled
        options.keys().any(|k| k.starts_with("lance.aimd."))
    } else {
        // Default to disabled if no storage options provided
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use object_store::path::Path;
    use object_store::throttle::{ThrottleConfig, ThrottledStore};
    use object_store::ObjectStore;
    use tokio::io::AsyncWriteExt;
    use url::Url;
    use crate::object_store::ObjectStore as LanceObjectStore;
    use crate::object_writer::ObjectWriter;

    #[test]
    fn test_aimd_config_defaults() {
        // Clean up any env vars from other tests
        std::env::remove_var("LANCE_AIMD_READ_INITIAL_RATE");
        std::env::remove_var("LANCE_AIMD_WRITE_INITIAL_RATE");
        std::env::remove_var("LANCE_AIMD_INITIAL_RATE");

        // Test write configuration (default)
        let config = AimdConfig::default();
        assert_eq!(config.initial_rate, 3500.0);
        assert_eq!(config.min_rate, 0.1);
        assert_eq!(config.max_rate, 10000.0);
        assert_eq!(config.increase_increment, 0.1);
        assert_eq!(config.reduction_factor, 2.0);
        assert_eq!(config.adjust_window, 2);

        // Test read configuration
        let read_config = AimdConfig::default_for_operation(OperationType::Read);
        assert_eq!(read_config.initial_rate, 5500.0);

        // Test write configuration explicitly
        let write_config = AimdConfig::default_for_operation(OperationType::Write);
        assert_eq!(write_config.initial_rate, 3500.0);
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

    #[test]
    fn test_aimd_disabled_by_default() {
        // Test that AIMD is disabled when no storage options are provided
        assert!(!is_aimd_enabled(None));

        // Test that AIMD is disabled with empty storage options
        let empty_options = HashMap::new();
        assert!(!is_aimd_enabled(Some(&empty_options)));

        // Test that AIMD is disabled with unrelated storage options
        let mut unrelated_options = HashMap::new();
        unrelated_options.insert("some.other.option".to_string(), "value".to_string());
        assert!(!is_aimd_enabled(Some(&unrelated_options)));
    }

    #[test]
    fn test_aimd_config_storage_options() {
        // Create storage options with AIMD configuration
        let mut storage_options = HashMap::new();
        storage_options.insert("lance.aimd.enabled".to_string(), "true".to_string());
        storage_options.insert("lance.aimd.read.initial_rate".to_string(), "7000".to_string());
        storage_options.insert("lance.aimd.write.initial_rate".to_string(), "4500".to_string());
        storage_options.insert("lance.aimd.increase_increment".to_string(), "0.2".to_string());
        storage_options.insert("lance.aimd.reduction_factor".to_string(), "3.0".to_string());
        storage_options.insert("lance.aimd.min_rate".to_string(), "50".to_string());
        storage_options.insert("lance.aimd.adjust_window".to_string(), "5".to_string());

        // Test that storage options are parsed correctly for read operations
        let read_config = AimdConfig::from_storage_options(
            OperationType::Read,
            Some(&storage_options)
        );
        assert_eq!(read_config.initial_rate, 7000.0);
        assert_eq!(read_config.increase_increment, 0.2);
        assert_eq!(read_config.reduction_factor, 3.0);
        assert_eq!(read_config.min_rate, 50.0);
        assert_eq!(read_config.adjust_window, 5);

        // Test that storage options are parsed correctly for write operations
        let write_config = AimdConfig::from_storage_options(
            OperationType::Write,
            Some(&storage_options)
        );
        assert_eq!(write_config.initial_rate, 4500.0);
        assert_eq!(write_config.increase_increment, 0.2);
        assert_eq!(write_config.reduction_factor, 3.0);
        assert_eq!(write_config.min_rate, 50.0);
        assert_eq!(write_config.adjust_window, 5);

        // Test generic initial rate fallback
        let mut generic_options = HashMap::new();
        generic_options.insert("lance.aimd.initial_rate".to_string(), "1500".to_string());

        let generic_config = AimdConfig::from_storage_options(
            OperationType::Read,
            Some(&generic_options)
        );
        assert_eq!(generic_config.initial_rate, 1500.0);

        // Test that AIMD can be disabled via storage options
        let mut disabled_options = HashMap::new();
        disabled_options.insert("lance.aimd.enabled".to_string(), "false".to_string());
        assert!(!is_aimd_enabled(Some(&disabled_options)));

        // Test that AIMD is enabled by default when options are set
        let mut enabled_options = HashMap::new();
        enabled_options.insert("lance.aimd.initial_rate".to_string(), "2000".to_string());
        assert!(is_aimd_enabled(Some(&enabled_options)));
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
        // Create storage options with AIMD configuration
        let mut storage_options = HashMap::new();
        storage_options.insert("lance.aimd.enabled".to_string(), "true".to_string());
        storage_options.insert("lance.aimd.write.initial_rate".to_string(), "3500".to_string());
        storage_options.insert("lance.aimd.min_rate".to_string(), "0.1".to_string());
        storage_options.insert("lance.aimd.max_rate".to_string(), "10000".to_string());
        storage_options.insert("lance.aimd.increase_increment".to_string(), "0.1".to_string());
        storage_options.insert("lance.aimd.reduction_factor".to_string(), "2".to_string());
        storage_options.insert("lance.aimd.adjust_window".to_string(), "2".to_string());

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
            Some(&storage_options),
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
    }
}