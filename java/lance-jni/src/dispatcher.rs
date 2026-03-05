// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use jni::JavaVM;
use jni::objects::GlobalRef;
use std::sync::{Arc, OnceLock};
use tokio::sync::mpsc;

/// Message sent from Tokio tasks to the dispatcher thread
pub struct DispatcherMessage {
    pub scanner_global_ref: GlobalRef,
    pub task_id: u64,
    pub result_ptr: i64, // >0: stream pointer, 0: EOF, <0: error
    pub error_msg: Option<String>,
}

/// Global dispatcher instance initialized in JNI_OnLoad
pub static DISPATCHER: OnceLock<Arc<Dispatcher>> = OnceLock::new();

/// Dispatcher manages a persistent JNI thread for completing Java futures
#[derive(Debug)]
pub struct Dispatcher {
    tx: mpsc::UnboundedSender<DispatcherMessage>,
}

impl Dispatcher {
    /// Initialize the dispatcher with a persistent JNI thread
    pub fn initialize(jvm: Arc<JavaVM>) -> Arc<Self> {
        let (tx, mut rx) = mpsc::unbounded_channel::<DispatcherMessage>();

        // Spawn persistent dispatcher thread
        std::thread::Builder::new()
            .name("lance-jni-dispatcher".to_string())
            .spawn(move || {
                // Attach ONCE and never detach - this is the key optimization
                let mut env = jvm
                    .attach_current_thread_permanently()
                    .expect("Failed to attach dispatcher to JVM");

                log::info!("JNI dispatcher thread started");

                // Cache method IDs for completeTask and failTask
                let async_scanner_class = env
                    .find_class("org/lance/ipc/AsyncScanner")
                    .expect("AsyncScanner class not found");
                let complete_method = env
                    .get_method_id(&async_scanner_class, "completeTask", "(JJ)V")
                    .expect("completeTask method not found");
                let fail_method = env
                    .get_method_id(&async_scanner_class, "failTask", "(JLjava/lang/String;)V")
                    .expect("failTask method not found");

                // Event loop: block waiting for completions
                while let Some(msg) = rx.blocking_recv() {
                    let scanner_obj = msg.scanner_global_ref.as_obj();

                    if let Some(error) = msg.error_msg {
                        // Error path
                        match env.new_string(&error) {
                            Ok(error_jstr) => {
                                let result = unsafe {
                                    env.call_method_unchecked(
                                        scanner_obj,
                                        fail_method,
                                        jni::signature::ReturnType::Primitive(
                                            jni::signature::Primitive::Void,
                                        ),
                                        &[
                                            jni::sys::jvalue {
                                                j: msg.task_id as i64,
                                            },
                                            jni::sys::jvalue {
                                                l: error_jstr.as_raw(),
                                            },
                                        ],
                                    )
                                };
                                if let Err(e) = result {
                                    log::error!("Failed to call failTask: {:?}", e);
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to create JString for error: {:?}", e);
                            }
                        }
                    } else {
                        // Success path
                        let result = unsafe {
                            env.call_method_unchecked(
                                scanner_obj,
                                complete_method,
                                jni::signature::ReturnType::Primitive(
                                    jni::signature::Primitive::Void,
                                ),
                                &[
                                    jni::sys::jvalue {
                                        j: msg.task_id as i64,
                                    },
                                    jni::sys::jvalue { j: msg.result_ptr },
                                ],
                            )
                        };
                        if let Err(e) = result {
                            log::error!("Failed to call completeTask: {:?}", e);
                        }
                    }
                }

                log::info!("JNI dispatcher thread shutting down");
            })
            .expect("Failed to spawn dispatcher thread");

        Arc::new(Self { tx })
    }

    /// Send a completion message to the dispatcher
    pub fn send(&self, msg: DispatcherMessage) -> std::result::Result<(), String> {
        self.tx
            .send(msg)
            .map_err(|e| format!("Failed to send message to dispatcher: {}", e))
    }
}
