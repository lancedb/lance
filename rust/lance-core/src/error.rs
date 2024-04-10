// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use arrow_schema::ArrowError;
use snafu::{Location, Snafu};

type BoxedError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub fn box_error(e: impl std::error::Error + Send + Sync + 'static) -> BoxedError {
    Box::new(e)
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("Invalid user input: {source}, {location}"))]
    InvalidInput {
        source: BoxedError,
        location: Location,
    },
    #[snafu(display("Dataset already exists: {uri}, {location}"))]
    DatasetAlreadyExists { uri: String, location: Location },
    #[snafu(display("Append with different schema: {difference}, location: {location}"))]
    SchemaMismatch {
        difference: String,
        location: Location,
    },
    #[snafu(display("Dataset at path {path} was not found: {source}, {location}"))]
    DatasetNotFound {
        path: String,
        source: BoxedError,
        location: Location,
    },
    #[snafu(display("Encountered corrupt file {path}: {source}, {location}"))]
    CorruptFile {
        path: object_store::path::Path,
        source: BoxedError,
        location: Location,
        // TODO: add backtrace?
    },
    #[snafu(display("Not supported: {source}, {location}"))]
    NotSupported {
        source: BoxedError,
        location: Location,
    },
    #[snafu(display("Commit conflict for version {version}: {source}, {location}"))]
    CommitConflict {
        version: u64,
        source: BoxedError,
        location: Location,
    },
    #[snafu(display("Encountered internal error. Please file a bug report at https://github.com/lancedb/lance/issues. {message}, {location}"))]
    Internal { message: String, location: Location },
    #[snafu(display("A prerequisite task failed: {message}, {location}"))]
    PrerequisiteFailed { message: String, location: Location },
    #[snafu(display("LanceError(Arrow): {message}, {location}"))]
    Arrow { message: String, location: Location },
    #[snafu(display("LanceError(Schema): {message}, {location}"))]
    Schema { message: String, location: Location },
    #[snafu(display("Not found: {uri}, {location}"))]
    NotFound { uri: String, location: Location },
    #[snafu(display("LanceError(IO): {message}, {location}"))]
    IO { message: String, location: Location },
    #[snafu(display("LanceError(Index): {message}, {location}"))]
    Index { message: String, location: Location },
    #[snafu(display("Lance index not found: {identity}, {location}"))]
    IndexNotFound {
        identity: String,
        location: Location,
    },
    #[snafu(display("Cannot infer storage location from: {message}"))]
    InvalidTableLocation { message: String },
    /// Stream early stop
    Stop,
    #[snafu(display("Wrapped error: {error}, {location}"))]
    Wrapped {
        error: BoxedError,
        location: Location,
    },
    #[snafu(display("Cloned error: {message}, {location}"))]
    Cloned { message: String, location: Location },
}

impl Error {
    pub fn corrupt_file(
        path: object_store::path::Path,
        message: impl Into<String>,
        location: Location,
    ) -> Self {
        let message: String = message.into();
        Self::CorruptFile {
            path,
            source: message.into(),
            location,
        }
    }

    pub fn invalid_input(message: impl Into<String>, location: Location) -> Self {
        let message: String = message.into();
        Self::InvalidInput {
            source: message.into(),
            location,
        }
    }
}

trait ToSnafuLocation {
    fn to_snafu_location(&'static self) -> snafu::Location;
}

impl ToSnafuLocation for std::panic::Location<'static> {
    fn to_snafu_location(&'static self) -> snafu::Location {
        snafu::Location::new(self.file(), self.line(), self.column())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<ArrowError> for Error {
    #[track_caller]
    fn from(e: ArrowError) -> Self {
        Self::Arrow {
            message: e.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<&ArrowError> for Error {
    #[track_caller]
    fn from(e: &ArrowError) -> Self {
        Self::Arrow {
            message: e.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<std::io::Error> for Error {
    #[track_caller]
    fn from(e: std::io::Error) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<object_store::Error> for Error {
    #[track_caller]
    fn from(e: object_store::Error) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<prost::DecodeError> for Error {
    #[track_caller]
    fn from(e: prost::DecodeError) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<prost::EncodeError> for Error {
    #[track_caller]
    fn from(e: prost::EncodeError) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<tokio::task::JoinError> for Error {
    #[track_caller]
    fn from(e: tokio::task::JoinError) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<object_store::path::Error> for Error {
    #[track_caller]
    fn from(e: object_store::path::Error) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<url::ParseError> for Error {
    #[track_caller]
    fn from(e: url::ParseError) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<serde_json::Error> for Error {
    #[track_caller]
    fn from(e: serde_json::Error) -> Self {
        Self::Arrow {
            message: e.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

#[track_caller]
fn arrow_io_error_from_msg(message: String) -> ArrowError {
    ArrowError::IoError(
        message.clone(),
        std::io::Error::new(std::io::ErrorKind::Other, message),
    )
}

impl From<Error> for ArrowError {
    fn from(value: Error) -> Self {
        match value {
            Error::Arrow { message, .. } => arrow_io_error_from_msg(message), // we lose the error type converting to LanceError
            Error::IO { message, .. } => arrow_io_error_from_msg(message),
            Error::Schema { message, .. } => Self::SchemaError(message),
            Error::Index { message, .. } => arrow_io_error_from_msg(message),
            Error::Stop => arrow_io_error_from_msg("early stop".to_string()),
            e => arrow_io_error_from_msg(e.to_string()), // Find a more scalable way of doing this
        }
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_sql::sqlparser::parser::ParserError> for Error {
    #[track_caller]
    fn from(e: datafusion_sql::sqlparser::parser::ParserError) -> Self {
        Self::IO {
            message: e.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_sql::sqlparser::tokenizer::TokenizerError> for Error {
    #[track_caller]
    fn from(e: datafusion_sql::sqlparser::tokenizer::TokenizerError) -> Self {
        Self::IO {
            message: e.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

#[cfg(feature = "datafusion")]
impl From<Error> for datafusion_common::DataFusionError {
    #[track_caller]
    fn from(e: Error) -> Self {
        Self::Execution(e.to_string())
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_common::DataFusionError> for Error {
    #[track_caller]
    fn from(e: datafusion_common::DataFusionError) -> Self {
        Self::IO {
            message: e.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

// This is a bit odd but some object_store functions only accept
// Stream<Result<T, ObjectStoreError>> and so we need to convert
// to ObjectStoreError to call the methods.
impl From<Error> for object_store::Error {
    fn from(err: Error) -> Self {
        Self::Generic {
            store: "N/A",
            source: Box::new(err),
        }
    }
}

#[track_caller]
pub fn get_caller_location() -> &'static std::panic::Location<'static> {
    std::panic::Location::caller()
}

/// Wrap an error in a new error type that implements Clone
///
/// This is useful when two threads/streams share a common fallible source
/// The base error will always have the full error.  Any cloned results will
/// only have Error::Cloned with the to_string of the base error.
pub struct CloneableError(pub Error);

impl Clone for CloneableError {
    #[track_caller]
    fn clone(&self) -> Self {
        Self(Error::Cloned {
            message: self.0.to_string(),
            location: std::panic::Location::caller().to_snafu_location(),
        })
    }
}

#[derive(Clone)]
pub struct CloneableResult<T: Clone>(pub std::result::Result<T, CloneableError>);

impl<T: Clone> From<Result<T>> for CloneableResult<T> {
    fn from(result: Result<T>) -> Self {
        Self(result.map_err(CloneableError))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_caller_location_capture() {
        let current_fn = get_caller_location();
        // make sure ? captures the correct location
        // .into() WILL NOT capture the correct location
        let f: Box<dyn Fn() -> Result<()>> = Box::new(|| {
            Err(object_store::Error::Generic {
                store: "",
                source: "".into(),
            })?;
            Ok(())
        });
        match f().unwrap_err() {
            Error::IO { location, .. } => {
                // +4 is the beginning of object_store::Error::Generic...
                assert_eq!(location.line, current_fn.line() + 4, "{}", location)
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected ObjectStore error"),
        }
    }
}
