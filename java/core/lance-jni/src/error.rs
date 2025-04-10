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

#[derive(Debug, PartialEq, Eq)]
pub enum JavaExceptionClass {
    IllegalArgumentException,
    IOException,
    RuntimeException,
    UnsupportedOperationException,
    AlreadyInException,
}

impl JavaExceptionClass {
    pub fn as_str(&self) -> &str {
        match self {
            Self::IllegalArgumentException => "java/lang/IllegalArgumentException",
            Self::IOException => "java/io/IOException",
            Self::RuntimeException => "java/lang/RuntimeException",
            Self::UnsupportedOperationException => "java/lang/UnsupportedOperationException",
            // Included for display purposes.  This is not a real exception.
            Self::AlreadyInException => "AlreadyInException",
        }
    }
}

#[derive(Debug)]
pub struct Error {
    message: String,
    java_class: JavaExceptionClass,
}

impl Error {
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

    pub fn in_exception() -> Self {
        Self {
            message: String::default(),
            java_class: JavaExceptionClass::AlreadyInException,
        }
    }

    pub fn throw(&self, env: &mut JNIEnv) {
        if self.java_class == JavaExceptionClass::AlreadyInException {
            // An exception is already in progress, so we don't need to throw another one.
            return;
        }
        if let Err(e) = env.throw_new(self.java_class.as_str(), &self.message) {
            eprintln!("Error when throwing Java exception: {:?}", e.to_string());
            panic!("Error when throwing Java exception: {:?}", e);
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.java_class.as_str(), self.message)
    }
}

impl From<LanceError> for Error {
    fn from(err: LanceError) -> Self {
        match err {
            LanceError::DatasetNotFound { .. }
            | LanceError::DatasetAlreadyExists { .. }
            | LanceError::CommitConflict { .. }
            | LanceError::InvalidInput { .. } => Self::input_error(err.to_string()),
            LanceError::IO { .. } => Self::io_error(err.to_string()),
            LanceError::NotSupported { .. } => Self::unsupported_error(err.to_string()),
            LanceError::NotFound { .. } => Self::io_error(err.to_string()),
            _ => Self::runtime_error(err.to_string()),
        }
    }
}

impl From<ArrowError> for Error {
    fn from(err: ArrowError) -> Self {
        match err {
            ArrowError::InvalidArgumentError { .. } => Self::input_error(err.to_string()),
            ArrowError::IoError { .. } => Self::io_error(err.to_string()),
            ArrowError::NotYetImplemented(_) => Self::unsupported_error(err.to_string()),
            _ => Self::runtime_error(err.to_string()),
        }
    }
}

impl From<JsonError> for Error {
    fn from(err: JsonError) -> Self {
        Self::io_error(err.to_string())
    }
}

impl From<JniError> for Error {
    fn from(err: JniError) -> Self {
        match err {
            // If we get this then it means that an exception was already in progress.  We can't
            // throw another one so we just return an error indicating that.
            JniError::JavaException => Self::in_exception(),
            _ => Self::runtime_error(err.to_string()),
        }
    }
}

impl From<Utf8Error> for Error {
    fn from(err: Utf8Error) -> Self {
        Self::input_error(err.to_string())
    }
}
