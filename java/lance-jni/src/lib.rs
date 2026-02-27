// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

#[macro_export]
macro_rules! ok_or_throw {
    ($env:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                err.throw(&mut $env);
                return JObject::null();
            }
        }
    };
}

macro_rules! ok_or_throw_without_return {
    ($env:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                err.throw(&mut $env);
                return;
            }
        }
    };
}

#[macro_export]
macro_rules! ok_or_throw_with_return {
    ($env:expr, $result:expr, $ret:expr) => {
        match $result {
            Ok(value) => value,
            Err(err) => {
                err.throw(&mut $env);
                return $ret;
            }
        }
    };
}

/// Extracts a panic message from a `catch_unwind` error payload.
pub fn panic_message(panic_info: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = panic_info.downcast_ref::<&str>() {
        format!("Rust panic: {}", s)
    } else if let Some(s) = panic_info.downcast_ref::<String>() {
        format!("Rust panic: {}", s)
    } else {
        "Rust panic: <unknown>".to_string()
    }
}

/// Wraps a JNI function body with `catch_unwind` to convert Rust panics into
/// Java `RuntimeException`s. Use for functions returning `JObject`.
#[macro_export]
macro_rules! jni_entry {
    ($env:expr, $body:expr) => {{
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        match result {
            Ok(val) => val,
            Err(panic_info) => {
                let msg = $crate::panic_message(&panic_info);
                let _ = $env.throw_new("java/lang/RuntimeException", &msg);
                JObject::null()
            }
        }
    }};
}

/// Wraps a JNI function body with `catch_unwind` for void functions.
#[macro_export]
macro_rules! jni_entry_void {
    ($env:expr, $body:expr) => {{
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        match result {
            Ok(val) => val,
            Err(panic_info) => {
                let msg = $crate::panic_message(&panic_info);
                let _ = $env.throw_new("java/lang/RuntimeException", &msg);
            }
        }
    }};
}

/// Wraps a JNI function body with `catch_unwind` for functions returning
/// a custom default value (e.g., `jlong`, `jint`, `jbyteArray`).
#[macro_export]
macro_rules! jni_entry_with_return {
    ($env:expr, $default:expr, $body:expr) => {{
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body));
        match result {
            Ok(val) => val,
            Err(panic_info) => {
                let msg = $crate::panic_message(&panic_info);
                let _ = $env.throw_new("java/lang/RuntimeException", &msg);
                $default
            }
        }
    }};
}

mod blocking_blob;
mod blocking_dataset;
mod blocking_scanner;
mod delta;
pub mod error;
pub mod ffi;
mod file_reader;
mod file_writer;
mod fragment;
mod index;
mod merge_insert;
mod namespace;
mod optimize;
mod schema;
mod session;
mod sql;
mod storage_options;
pub mod traits;
mod transaction;
pub mod utils;
mod vector_trainer;

pub use error::Error;
pub use error::Result;
pub use ffi::JNIEnvExt;
pub use storage_options::JavaStorageOptionsProvider;

use env_logger::{Builder, Env};
use std::env;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::Arc;

use std::sync::LazyLock;

pub static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

fn set_timestamp_precision(builder: &mut env_logger::Builder) {
    if let Ok(timestamp_precision) = env::var("LANCE_LOG_TS_PRECISION") {
        match timestamp_precision.as_str() {
            "ns" => {
                builder.format_timestamp_nanos();
            }
            "us" => {
                builder.format_timestamp_micros();
            }
            "ms" => {
                builder.format_timestamp_millis();
            }
            "s" => {
                builder.format_timestamp_secs();
            }
            _ => {
                // Can't log here because logging is not initialized yet
                println!(
                    "Invalid timestamp precision (valid values: ns, us, ms, s): {}, using default",
                    timestamp_precision
                );
            }
        };
    }
}

fn set_log_file_target(builder: &mut env_logger::Builder) {
    if let Ok(log_file_path) = env::var("LANCE_LOG_FILE") {
        let path = Path::new(&log_file_path);

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                println!(
                    "Failed to create parent directories for log file '{}': {}, using stderr",
                    log_file_path, e
                );
                return;
            }
        }

        // Try to open/create the log file
        match OpenOptions::new().create(true).append(true).open(path) {
            Ok(file) => {
                builder.target(env_logger::Target::Pipe(Box::new(file)));
            }
            Err(e) => {
                println!(
                    "Failed to open log file '{}': {}, using stderr",
                    log_file_path, e
                );
            }
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_org_lance_JniLoader_initLanceLogger() {
    // This JNI function has no JNIEnv parameter, so we catch panics without
    // throwing Java exceptions. The panic hook (installed below) will log the
    // message.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let env = Env::new()
            .filter_or("LANCE_LOG", "warn")
            .write_style("LANCE_LOG_STYLE");
        let mut log_builder = Builder::from_env(env);
        set_timestamp_precision(&mut log_builder);
        set_log_file_target(&mut log_builder);
        let logger = Arc::new(log_builder.build());
        let max_level = logger.filter();
        // Ignore error if logger was already set (e.g., called multiple times)
        let _ = log::set_boxed_logger(Box::new(logger.clone()));
        log::set_max_level(max_level);

        std::panic::set_hook(Box::new(|info| {
            let msg = if let Some(s) = info.payload().downcast_ref::<&str>() {
                s.to_string()
            } else if let Some(s) = info.payload().downcast_ref::<String>() {
                s.clone()
            } else {
                "<unknown>".to_string()
            };
            let location = info
                .location()
                .map(|l| format!(" at {}:{}:{}", l.file(), l.line(), l.column()))
                .unwrap_or_default();
            log::error!("Rust panic{}: {}", location, msg);
        }));
    }));
}
