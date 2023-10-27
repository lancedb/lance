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
use snafu::{location, Location, Snafu};

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
    // #[snafu(display("Append with different schema: original={original} new={new}"))]
    #[snafu(display("Append with different schema:"))]
    SchemaMismatch {},
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
    #[snafu(display("A prerequisite task failed: {message}"))]
    PrerequisiteFailed { message: String },
    #[snafu(display("LanceError(Arrow): {message}"))]
    Arrow { message: String },
    #[snafu(display("LanceError(Schema): {message}, {location}"))]
    Schema { message: String, location: Location },
    #[snafu(display("Not found: {uri}, {location}"))]
    NotFound { uri: String, location: Location },
    #[snafu(display("LanceError(IO): {message}, {location}"))]
    IO { message: String, location: Location },
    #[snafu(display("LanceError(Index): {message}"))]
    Index { message: String },
    /// Stream early stop
    Stop,
}

impl Error {
    pub fn corrupt_file(path: object_store::path::Path, message: impl Into<String>) -> Self {
        let message: String = message.into();
        Self::CorruptFile {
            path,
            source: message.into(),
            location: location!(),
        }
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        let message: String = message.into();
        Self::InvalidInput {
            source: message.into(),
            location: location!(),
        }
    }
}

pub type Result<T> = std::result::Result<T, Error>;

impl From<ArrowError> for Error {
    fn from(e: ArrowError) -> Self {
        Self::Arrow {
            message: e.to_string(),
        }
    }
}

impl From<&ArrowError> for Error {
    fn from(e: &ArrowError) -> Self {
        Self::Arrow {
            message: e.to_string(),
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: location!(),
        }
    }
}

impl From<object_store::Error> for Error {
    fn from(e: object_store::Error) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: location!(),
        }
    }
}

impl From<prost::DecodeError> for Error {
    fn from(e: prost::DecodeError) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: location!(),
        }
    }
}

impl From<tokio::task::JoinError> for Error {
    fn from(e: tokio::task::JoinError) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: location!(),
        }
    }
}

impl From<object_store::path::Error> for Error {
    fn from(e: object_store::path::Error) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: location!(),
        }
    }
}

impl From<url::ParseError> for Error {
    fn from(e: url::ParseError) -> Self {
        Self::IO {
            message: (e.to_string()),
            location: location!(),
        }
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Self::Arrow {
            message: e.to_string(),
        }
    }
}

fn arrow_io_error_from_msg(message: String) -> ArrowError {
    ArrowError::IoError(
        message.clone(),
        std::io::Error::new(std::io::ErrorKind::Other, message),
    )
}

impl From<Error> for ArrowError {
    fn from(value: Error) -> Self {
        match value {
            Error::Arrow { message } => arrow_io_error_from_msg(message), // we lose the error type converting to LanceError
            Error::IO { message, .. } => arrow_io_error_from_msg(message),
            Error::Schema { message, .. } => Self::SchemaError(message),
            Error::Index { message } => arrow_io_error_from_msg(message),
            Error::Stop => arrow_io_error_from_msg("early stop".to_string()),
            e => arrow_io_error_from_msg(e.to_string()), // Find a more scalable way of doing this
        }
    }
}

impl From<datafusion_sql::sqlparser::parser::ParserError> for Error {
    fn from(e: datafusion_sql::sqlparser::parser::ParserError) -> Self {
        Self::IO {
            message: e.to_string(),
            location: location!(),
        }
    }
}

impl From<datafusion_sql::sqlparser::tokenizer::TokenizerError> for Error {
    fn from(e: datafusion_sql::sqlparser::tokenizer::TokenizerError) -> Self {
        Self::IO {
            message: e.to_string(),
            location: location!(),
        }
    }
}

impl From<Error> for datafusion_common::DataFusionError {
    fn from(e: Error) -> Self {
        Self::Execution(e.to_string())
    }
}

impl From<datafusion_common::DataFusionError> for Error {
    fn from(e: datafusion_common::DataFusionError) -> Self {
        Self::IO {
            message: e.to_string(),
            location: location!(),
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
