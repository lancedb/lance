use rand::{Rng, SeedableRng};
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

    pub fn with_unit(self, unit: u32) -> Self {
        Self { unit, ..self }
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

/// SlotBackoff is a backoff strategy that uses a slot-based approach.
///
/// This is for when the cause of the failure is concurrency itself and that
/// the attempts take about the same amount of time.
///
/// Say you have N attempts to do something. We don't know there are N ahead of
/// time. We start guessing with 4 slots:
///
/// | 1, 2, 3 | 4, 5, 6 | 7, 8, 9 | 10 |
///
/// Each slot can have one success, so we can eliminate 3, 6, 9, and 10. In the
/// next round, we will use twice as many slots (8):
///
/// | 1 | 2 | 4 | 5 | 7 | 8 | ... |
///
/// Optimally, that should be 16 total attempts. For a 1s unit, the retry times
/// are:
/// * Round 1: 0 - 3s
/// * Round 2: 0 - 7s
/// * Round 3: 0 - 15s
/// * ...
pub struct SlotBackoff {
    base: u32,
    unit: u32,
    starting_i: u32,
    attempt: u32,
    rng: rand::rngs::SmallRng,
}

impl Default for SlotBackoff {
    fn default() -> Self {
        Self {
            base: 2,
            unit: 50,
            starting_i: 2, // start with 4 slots
            attempt: 0,
            rng: rand::rngs::SmallRng::from_entropy(),
        }
    }
}

impl SlotBackoff {
    pub fn with_unit(self, unit: u32) -> Self {
        Self { unit, ..self }
    }

    pub fn attempt(&self) -> u32 {
        self.attempt
    }

    pub fn next_backoff(&mut self) -> Duration {
        let num_slots = self.base.saturating_pow(self.attempt + self.starting_i);
        let slot_i = self.rng.gen_range(0..num_slots);
        self.attempt += 1;
        Duration::from_millis((slot_i * self.unit) as u64)
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

    #[test]
    fn test_slot_backoff() {
        fn assert_in(value: u128, expected: &[u128]) {
            assert!(
                expected.iter().any(|&x| x == value),
                "value {} not in {:?}",
                value,
                expected
            );
        }

        for _ in 0..10 {
            let mut backoff = SlotBackoff::default().with_unit(100);
            assert_in(backoff.next_backoff().as_millis(), &[0, 100, 200, 300]);
            assert_eq!(backoff.attempt(), 1);
            assert_in(
                backoff.next_backoff().as_millis(),
                &[0, 100, 200, 300, 400, 500, 600, 700],
            );
            assert_eq!(backoff.attempt(), 2);
            assert_in(
                backoff.next_backoff().as_millis(),
                &(0..16).map(|i| i * 100).collect::<Vec<_>>(),
            );
            assert_eq!(backoff.attempt(), 3);
        }
    }
}
