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

mod blocking_dataset;
mod blocking_scanner;
pub mod error;
pub mod ffi;
mod file_reader;
mod file_writer;
mod fragment;
mod schema;
mod sql;
pub mod traits;
mod transaction;
pub mod utils;

pub use error::Error;
pub use error::Result;
pub use ffi::JNIEnvExt;

use env_logger::{Builder, Env};
use std::sync::Arc;

use std::sync::LazyLock;

pub static RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime")
});

#[no_mangle]
pub extern "system" fn Java_com_lancedb_lance_JniLoader_initLanceLogger() {
    let env = Env::new()
        .filter_or("LANCE_LOG", "warn")
        .write_style("LANCE_LOG_STYLE");
    let mut log_builder = Builder::from_env(env);
    let logger = Arc::new(log_builder.build());
    let max_level = logger.filter();
    log::set_boxed_logger(Box::new(logger.clone())).unwrap();
    log::set_max_level(max_level);
    // todo: add tracing
}
