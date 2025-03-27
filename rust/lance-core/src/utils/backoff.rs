use rand::Rng;
use std::time::Duration;

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

/// Computes backoff as
///
/// ```text
/// backoff = base^attempt * unit + jitter
/// ```
///
/// The defaults are base=2, unit=50ms, jitter=50ms, min=0ms, max=5s. This gives
/// a backoff of 50ms, 100ms, 200ms, 400ms, 800ms, 1.6s, 3.2s, 5s, (not including jitter).
///
/// You can have non-exponential backoff by setting base=1.
pub struct Backoff {
    base: u32,
    unit: u32,
    jitter: i32,
    min: u32,
    max: u32,
    attempt: u32,
}

impl Default for Backoff {
    fn default() -> Self {
        Self {
            base: 2,
            unit: 50,
            jitter: 50,
            min: 0,
            max: 5000,
            attempt: 0,
        }
    }
}

impl Backoff {
    pub fn with_base(self, base: u32) -> Self {
        Self { base, ..self }
    }

    pub fn with_jitter(self, jitter: i32) -> Self {
        Self { jitter, ..self }
    }

    pub fn with_min(self, min: u32) -> Self {
        Self { min, ..self }
    }

    pub fn with_max(self, max: u32) -> Self {
        Self { max, ..self }
    }

    pub fn next_backoff(&mut self) -> Duration {
        let backoff = self
            .base
            .saturating_pow(self.attempt)
            .saturating_mul(self.unit);
        let jitter = rand::thread_rng().gen_range(-self.jitter..=self.jitter);
        let backoff = (backoff.saturating_add_signed(jitter)).clamp(self.min, self.max);
        self.attempt += 1;
        Duration::from_millis(backoff as u64)
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    pub fn reset(&mut self) {
        self.attempt = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backoff() {
        let mut backoff = Backoff::default().with_jitter(0);
        assert_eq!(backoff.next_backoff().as_millis(), 50);
        assert_eq!(backoff.attempt(), 1);
        assert_eq!(backoff.next_backoff().as_millis(), 100);
        assert_eq!(backoff.attempt(), 2);
        assert_eq!(backoff.next_backoff().as_millis(), 200);
        assert_eq!(backoff.attempt(), 3);
        assert_eq!(backoff.next_backoff().as_millis(), 400);
        assert_eq!(backoff.attempt(), 4);
    }
}
