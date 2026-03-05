// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use jni::objects::GlobalRef;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};
use tokio::sync::RwLock;

pub type TaskId = u64;

/// Information about an in-flight async task
pub struct TaskInfo {
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
}

/// Global task tracker instance
pub static TASK_TRACKER: LazyLock<TaskTracker> = LazyLock::new(TaskTracker::new);
