// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Progress tracking for distributed index building
#[derive(Debug, Clone)]
pub struct BuildProgress {
    pub phase: BuildPhase,
    pub current_fragment: usize,
    pub total_fragments: usize,
    pub processed_vectors: usize,
    pub total_vectors: usize,
    pub start_time: Instant,
    pub estimated_remaining: Duration,
    pub phase_progress: HashMap<String, f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BuildPhase {
    IvfTraining,
    HnswBuilding,
    IndexMerging,
    QualityValidation,
    Completed,
}

impl BuildProgress {
    pub fn new(total_fragments: usize, total_vectors: usize) -> Self {
        Self {
            phase: BuildPhase::IvfTraining,
            current_fragment: 0,
            total_fragments,
            processed_vectors: 0,
            total_vectors,
            start_time: Instant::now(),
            estimated_remaining: Duration::from_secs(0),
            phase_progress: HashMap::new(),
        }
    }

    pub fn update_phase(&mut self, phase: BuildPhase) {
        self.phase = phase;
    }

    pub fn update_fragment_progress(&mut self, fragment: usize, vectors: usize) {
        self.current_fragment = fragment;
        self.processed_vectors = vectors;
        self.update_estimated_remaining();
    }

    pub fn update_phase_progress(&mut self, phase_name: String, progress: f32) {
        self.phase_progress.insert(phase_name, progress);
    }

    fn update_estimated_remaining(&mut self) {
        let elapsed = self.start_time.elapsed();
        let progress_ratio = self.processed_vectors as f64 / self.total_vectors as f64;

        if progress_ratio > 0.0 {
            let total_estimated = elapsed.div_f64(progress_ratio);
            self.estimated_remaining = total_estimated.saturating_sub(elapsed);
        }
    }

    pub fn overall_progress(&self) -> f32 {
        let base_progress = match self.phase {
            BuildPhase::IvfTraining => 0.0,
            BuildPhase::HnswBuilding => 0.3,
            BuildPhase::IndexMerging => 0.7,
            BuildPhase::QualityValidation => 0.9,
            BuildPhase::Completed => 1.0,
        };

        let phase_multiplier = match self.phase {
            BuildPhase::IvfTraining => 0.3,
            BuildPhase::HnswBuilding => 0.4,
            BuildPhase::IndexMerging => 0.2,
            BuildPhase::QualityValidation => 0.1,
            BuildPhase::Completed => 0.0,
        };

        let fragment_progress = self.current_fragment as f32 / self.total_fragments as f32;
        base_progress + (fragment_progress * phase_multiplier)
    }

    pub fn format_string(&self) -> String {
        let elapsed = self.start_time.elapsed();
        format!(
            "Progress: {:.1}% | Phase: {:?} | Fragment: {}/{} | Vectors: {}/{} | Elapsed: {:?} | ETA: {:?}",
            self.overall_progress() * 100.0,
            self.phase,
            self.current_fragment,
            self.total_fragments,
            self.processed_vectors,
            self.total_vectors,
            elapsed,
            self.estimated_remaining
        )
    }
}

/// Thread-safe progress tracker
pub struct ProgressTracker {
    progress: Arc<Mutex<BuildProgress>>,
}

impl std::fmt::Display for BuildProgress {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elapsed = self.start_time.elapsed();
        write!(
            f,
            "Progress: {:.1}% | Phase: {:?} | Fragment: {}/{} | Vectors: {}/{} | Elapsed: {:?} | ETA: {:?}",
            self.overall_progress() * 100.0,
            self.phase,
            self.current_fragment,
            self.total_fragments,
            self.processed_vectors,
            self.total_vectors,
            elapsed,
            self.estimated_remaining
        )
    }
}

impl ProgressTracker {
    pub fn new(total_fragments: usize, total_vectors: usize) -> Self {
        Self {
            progress: Arc::new(Mutex::new(BuildProgress::new(
                total_fragments,
                total_vectors,
            ))),
        }
    }

    pub fn update_phase(&self, phase: BuildPhase) {
        let mut progress = self.progress.lock().unwrap();
        progress.update_phase(phase);
    }

    pub fn update_fragment_progress(&self, fragment: usize, vectors: usize) {
        let mut progress = self.progress.lock().unwrap();
        progress.update_fragment_progress(fragment, vectors);
    }

    pub fn update_phase_progress(&self, phase_name: String, progress_value: f32) {
        let mut progress = self.progress.lock().unwrap();
        progress.update_phase_progress(phase_name, progress_value);
    }

    pub fn get_progress(&self) -> BuildProgress {
        let progress = self.progress.lock().unwrap();
        progress.clone()
    }

    pub fn mark_completed(&self) {
        let mut progress = self.progress.lock().unwrap();
        progress.update_phase(BuildPhase::Completed);
    }
}
