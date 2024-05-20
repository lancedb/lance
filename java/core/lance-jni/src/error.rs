// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::str::Utf8Error;

use arrow_schema::ArrowError;
use jni::{errors::Error as JniError, JNIEnv};
use lance::error::Error as LanceError;
use serde_json::Error as JsonError;

pub type JavaResult<T> = std::result::Result<T, JavaError>;

#[derive(Debug)]
pub enum JavaExceptionClass {
    IllegalArgumentException,
    IOException,
    RuntimeException,
    UnsupportedOperationException,
}

impl JavaExceptionClass {
    pub fn as_str(&self) -> &str {
        match self {
            Self::IllegalArgumentException => "java/lang/IllegalArgumentException",
            Self::IOException => "java/io/IOException",
            Self::RuntimeException => "java/lang/RuntimeException",
            Self::UnsupportedOperationException => "java/lang/UnsupportedOperationException",
        }
    }
}

#[derive(Debug)]
pub struct JavaError {
    message: String,
    java_class: JavaExceptionClass,
}

impl JavaError {
    pub fn new(message: String, java_class: JavaExceptionClass) -> Self {
        Self {
            message,
            java_class,
        }
    }

    pub fn runtime_error(message: String) -> Self {
        Self {
            message,
            java_class: JavaExceptionClass::RuntimeException,
        }
    }

    pub fn io_error(message: String) -> Self {
        Self::new(message, JavaExceptionClass::IOException)
    }

    pub fn input_error(message: String) -> Self {
        Self::new(message, JavaExceptionClass::IllegalArgumentException)
    }

    pub fn unsupported_error(message: String) -> Self {
        Self::new(message, JavaExceptionClass::UnsupportedOperationException)
    }

    pub fn throw(&self, env: &mut JNIEnv) {
        env.throw_new(self.java_class.as_str(), &self.message)
            .expect("Error when throwing Java exception");
    }
}

impl std::fmt::Display for JavaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.java_class.as_str(), self.message)
    }
}

impl std::error::Error for JavaError {}

/// Trait for converting errors to Java exceptions.
pub trait JavaErrorConversion<T> {
    /// Convert to `JavaError` as I/O exception.
    fn io_error(self) -> JavaResult<T>;

    /// Convert to `JavaError` as runtime exception.
    fn runtime_error(self) -> JavaResult<T>;

    /// Convert to `JavaError` as value (input) exception.
    fn input_error(self) -> JavaResult<T>;

    /// Convert to `JavaError` as unsupported operation exception.
    fn unsupported_error(self) -> JavaResult<T>;
}

impl<T, E: std::fmt::Display> JavaErrorConversion<T> for std::result::Result<T, E> {
    fn io_error(self) -> JavaResult<T> {
        self.map_err(|err| JavaError::io_error(err.to_string()))
    }

    fn runtime_error(self) -> JavaResult<T> {
        self.map_err(|err| JavaError::runtime_error(err.to_string()))
    }

    fn input_error(self) -> JavaResult<T> {
        self.map_err(|err| JavaError::input_error(err.to_string()))
    }

    fn unsupported_error(self) -> JavaResult<T> {
        self.map_err(|err| JavaError::unsupported_error(err.to_string()))
    }
}

/// JavaErrorExt trait that converts specific error types to Java exceptions
pub trait JavaErrorExt<T> {
    /// Convert to a Java error based on the specific error type
    fn infer_error(self) -> JavaResult<T>;
}

impl<T> JavaErrorExt<T> for std::result::Result<T, LanceError> {
    fn infer_error(self) -> JavaResult<T> {
        match &self {
            Ok(_) => Ok(self.unwrap()),
            Err(err) => match err {
                LanceError::DatasetNotFound { .. }
                | LanceError::DatasetAlreadyExists { .. }
                | LanceError::CommitConflict { .. }
                | LanceError::InvalidInput { .. } => self.input_error(),
                LanceError::IO { .. } => self.io_error(),
                LanceError::NotSupported { .. } => self.unsupported_error(),
                _ => self.runtime_error(),
            },
        }
    }
}

impl<T> JavaErrorExt<T> for std::result::Result<T, ArrowError> {
    fn infer_error(self) -> JavaResult<T> {
        match &self {
            Ok(_) => Ok(self.unwrap()),
            Err(err) => match err {
                ArrowError::InvalidArgumentError { .. } => self.input_error(),
                ArrowError::IoError { .. } => self.io_error(),
                ArrowError::NotYetImplemented(_) => self.unsupported_error(),
                _ => self.runtime_error(),
            },
        }
    }
}

impl<T> JavaErrorExt<T> for std::result::Result<T, JsonError> {
    fn infer_error(self) -> JavaResult<T> {
        match &self {
            Ok(_) => Ok(self.unwrap()),
            Err(_) => self.io_error(),
        }
    }
}

impl<T> JavaErrorExt<T> for std::result::Result<T, JniError> {
    fn infer_error(self) -> JavaResult<T> {
        match &self {
            Ok(_) => Ok(self.unwrap()),
            Err(_) => self.runtime_error(),
        }
    }
}

impl<T> JavaErrorExt<T> for std::result::Result<T, Utf8Error> {
    fn infer_error(self) -> JavaResult<T> {
        match &self {
            Ok(_) => Ok(self.unwrap()),
            Err(_) => self.input_error(),
        }
    }
}
