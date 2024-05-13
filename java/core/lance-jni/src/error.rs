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
use jni::errors::Error as JniError;
use serde_json::Error as JsonError;
use snafu::{Location, Snafu};

/// Java Exception types
pub enum JavaException {
    IllegalArgumentException,
    IOException,
    RuntimeException,
}

impl JavaException {
    pub fn as_str(&self) -> &str {
        match self {
            Self::IllegalArgumentException => "java/lang/IllegalArgumentException",
            Self::IOException => "java/io/IOException",
            Self::RuntimeException => "java/lang/RuntimeException",
        }
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("JNI error: {message}, {location}"))]
    Jni { message: String, location: Location },
    #[snafu(display("Invalid argument: {message}, {location}"))]
    InvalidArgument { message: String, location: Location },
    #[snafu(display("IO error: {message}, {location}"))]
    IO { message: String, location: Location },
    #[snafu(display("Arrow error: {message}, {location}"))]
    Arrow { message: String, location: Location },
    #[snafu(display("Index error: {message}, {location}"))]
    Index { message: String, location: Location },
    #[snafu(display("JSON error: {message}, {location}"))]
    JSON { message: String, location: Location },
    #[snafu(display("Dataset at path {path} was not found, {location}"))]
    DatasetNotFound { path: String, location: Location },
    #[snafu(display("Dataset already exists: {uri}, {location}"))]
    DatasetAlreadyExists { uri: String, location: Location },
    #[snafu(display("Unknown error: {message}, {location}"))]
    Other { message: String, location: Location },
}

impl Error {
    /// Throw as Java Exception
    pub fn throw(&self, env: &mut jni::JNIEnv) {
        match self {
            Self::InvalidArgument { .. } => {
                self.throw_as(env, JavaException::IllegalArgumentException)
            }
            Self::IO { .. } | Self::Index { .. } => self.throw_as(env, JavaException::IOException),
            Self::Arrow { .. }
            | Self::DatasetNotFound { .. }
            | Self::DatasetAlreadyExists { .. }
            | Self::JSON { .. }
            | Self::Other { .. }
            | Self::Jni { .. } => self.throw_as(env, JavaException::RuntimeException),
        }
    }

    /// Throw as an concrete Java Exception
    pub fn throw_as(&self, env: &mut jni::JNIEnv, exception: JavaException) {
        env.throw_new(exception.as_str(), self.to_string())
            .expect("Error when throwing Java exception");
    }
}

pub type Result<T> = std::result::Result<T, Error>;

trait ToSnafuLocation {
    fn to_snafu_location(&'static self) -> snafu::Location;
}

impl ToSnafuLocation for std::panic::Location<'static> {
    fn to_snafu_location(&'static self) -> snafu::Location {
        snafu::Location::new(self.file(), self.line(), self.column())
    }
}

impl From<JniError> for Error {
    #[track_caller]
    fn from(source: JniError) -> Self {
        Self::Jni {
            message: source.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<Utf8Error> for Error {
    #[track_caller]
    fn from(source: Utf8Error) -> Self {
        Self::InvalidArgument {
            message: source.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<ArrowError> for Error {
    #[track_caller]
    fn from(source: ArrowError) -> Self {
        Self::Arrow {
            message: source.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<JsonError> for Error {
    #[track_caller]
    fn from(source: JsonError) -> Self {
        Self::JSON {
            message: source.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<lance::Error> for Error {
    #[track_caller]
    fn from(source: lance::Error) -> Self {
        match source {
            lance::Error::DatasetNotFound {
                path,
                source: _,
                location,
            } => Self::DatasetNotFound { path, location },
            lance::Error::DatasetAlreadyExists { uri, location } => {
                Self::DatasetAlreadyExists { uri, location }
            }
            lance::Error::IO { source, location } => Self::IO {
                message: source.to_string(),
                location,
            },
            lance::Error::Arrow { message, location } => Self::Arrow { message, location },
            lance::Error::Index { message, location } => Self::Index { message, location },
            lance::Error::InvalidInput { source, location } => Self::InvalidArgument {
                message: source.to_string(),
                location,
            },
            _ => Self::Other {
                message: source.to_string(),
                location: std::panic::Location::caller().to_snafu_location(),
            },
        }
    }
}
