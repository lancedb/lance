// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use jni::objects::GlobalRef;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use tokio::sync::RwLock;

pub type TaskId = u64;

/// Information about an in-flight async task
pub struct TaskInfo {
    #[allow(dead_code)] // Used for cleanup when task is cancelled
    pub scanner_global_ref: GlobalRef,
    pub cancel_handle: tokio::task::JoinHandle<()>,
}

/// Thread-safe task registry for managing async scan operations
pub struct TaskTracker {
    tasks: Arc<RwLock<HashMap<TaskId, TaskInfo>>>,
}

impl TaskTracker {
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new task
    pub async fn register(&self, task_id: TaskId, info: TaskInfo) {
        let mut tasks = self.tasks.write().await;
        tasks.insert(task_id, info);
    }

    /// Mark a task as complete and return its info
    pub async fn complete(&self, task_id: TaskId) -> Option<TaskInfo> {
        let mut tasks = self.tasks.write().await;
        tasks.remove(&task_id)
    }

    /// Cancel a task by ID
    pub async fn cancel(&self, task_id: TaskId) {
        let info = {
            let mut tasks = self.tasks.write().await;
            tasks.remove(&task_id)
        };

        if let Some(info) = info {
            info.cancel_handle.abort();
        }
    }

    // TODO: Implement timeout-based cleanup for defense-in-depth
    //
    // While TaskCleanupGuard (RAII pattern) ensures cleanup in normal and panic cases,
    // a background cleanup task provides additional safety against edge cases:
    //
    // Proposed implementation:
    // ```
    // pub async fn cleanup_stale_tasks(&self, max_age: Duration) {
    //     let mut tasks = self.tasks.write().await;
    //     let now = Instant::now();
    //     tasks.retain(|task_id, info| {
    //         let is_finished = info.cancel_handle.is_finished();
    //         let is_stale = info.created_at.elapsed() > max_age;
    //
    //         if is_finished || is_stale {
    //             log::warn!("Cleaning up stale/finished task {}", task_id);
    //             false // remove from HashMap
    //         } else {
    //             true // keep in HashMap
    //         }
    //     });
    // }
    //
    // // In JNI_OnLoad or module initialization:
    // RT.spawn(async {
    //     loop {
    //         tokio::time::sleep(Duration::from_secs(60)).await;
    //         TASK_TRACKER.cleanup_stale_tasks(Duration::from_secs(300)).await;
    //     }
    // });
    // ```
    //
    // This would require adding `created_at: Instant` field to TaskInfo.
}

/// Global task tracker instance
pub static TASK_TRACKER: LazyLock<TaskTracker> = LazyLock::new(TaskTracker::new);
