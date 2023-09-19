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

use crate::datatypes::Schema;

type BoxedError = Box<dyn std::error::Error + Send + Sync + 'static>;

pub(crate) fn box_error(e: impl std::error::Error + Send + Sync + 'static) -> BoxedError {
    Box::new(e)
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum Error {
    #[snafu(display("Invalid user input: {source}"))]
    InvalidInput { source: BoxedError },
    #[snafu(display("Dataset already exists: {uri}"))]
    DatasetAlreadyExists { uri: String },
    #[snafu(display("Append with different schema: original={original} new={new}"))]
    SchemaMismatch { original: Schema, new: Schema },
    #[snafu(display("Dataset at path {path} was not found: {source}"))]
    DatasetNotFound { path: String, source: BoxedError },
    #[snafu(display("Encountered corrupt file {path}: {source}"))]
    CorruptFile {
        path: object_store::path::Path,
        source: BoxedError,
        // TODO: add backtrace?
    },
    #[snafu(display("Not supported: {source}"))]
    NotSupported { source: BoxedError },
    #[snafu(display("Commit conflict for version {version}: {source}"))]
    CommitConflict { version: u64, source: BoxedError },
    #[snafu(display("Encountered internal error. Please file a bug report at https://github.com/lancedb/lance/issues. {message}"))]
    Internal { message: String },
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
        }
    }

    pub fn invalid_input(message: impl Into<String>) -> Self {
        let message: String = message.into();
        Self::InvalidInput {
            source: message.into(),
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

impl From<Error> for ArrowError {
    fn from(value: Error) -> Self {
        match value {
            Error::Arrow { message: err } => Self::IoError(err), // we lose the error type converting to LanceError
            Error::IO { message: err, .. } => Self::IoError(err),
            Error::Schema { message: err, .. } => Self::SchemaError(err),
            Error::Index { message: err } => Self::IoError(err),
            Error::Stop => Self::IoError("early stop".to_string()),
            e => Self::IoError(e.to_string()), // Find a more scalable way of doing this
        }
    }
}

impl From<datafusion::sql::sqlparser::parser::ParserError> for Error {
    fn from(e: datafusion::sql::sqlparser::parser::ParserError) -> Self {
        Self::IO {
            message: e.to_string(),
            location: location!(),
        }
    }
}

impl From<datafusion::sql::sqlparser::tokenizer::TokenizerError> for Error {
    fn from(e: datafusion::sql::sqlparser::tokenizer::TokenizerError) -> Self {
        Self::IO {
            message: e.to_string(),
            location: location!(),
        }
    }
}

impl From<Error> for datafusion::error::DataFusionError {
    fn from(e: Error) -> Self {
        Self::Execution(e.to_string())
    }
}

impl From<datafusion::error::DataFusionError> for Error {
    fn from(e: datafusion::error::DataFusionError) -> Self {
        Self::IO {
            message: e.to_string(),
            location: location!(),
        }
    }
}
