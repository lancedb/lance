// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::{
    backtrace::Backtrace,
    ops::{Deref, DerefMut},
};

use arrow_schema::ArrowError;
use http::StatusCode;

type BoxedError = Box<dyn std::error::Error + Send + Sync + 'static>;

// To keep the size of Result<T> small, we use a Box<InnerError> to store the
// actual error. This way, the size of Result<T> is just a pointer.
pub struct Error(pub Box<InnerError>);

impl Error {
    pub fn new(status: StatusCode, title: &'static str, details: impl Into<String>) -> Self {
        // TODO: automatically capture backtrace for 500 status?
        Self(Box::new(InnerError {
            title,
            details: details.into(),
            status_code: status,
            cause: None,
            backtrace: MaybeBacktrace::None,
        }))
    }

    pub fn bad_request(title: &'static str, details: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, title, details)
    }

    pub fn unexpected(title: &'static str, details: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, title, details)
    }

    pub fn with_cause(mut self, cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.0.cause = Some(Box::new(cause));
        self
    }

    pub fn with_backtrace(mut self) -> Self {
        self.0.backtrace = MaybeBacktrace::Captured(Backtrace::capture());
        self
    }

    #[track_caller]
    pub fn with_location(mut self) -> Self {
        self.0.backtrace = MaybeBacktrace::Location(std::panic::Location::caller());
        self
    }
}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            // Include backtrace and cause
            todo!()
        } else {
            // Just include stats, title, details
            write!(f, "{}: {}", self.title, self.details)
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self) // Re-use Debug impl
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.cause.as_ref().map(|e| e.as_ref() as _)
    }
}

impl Deref for Error {
    type Target = InnerError;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Error {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Lance's generic error.
///
/// The error aims to make it easy to determine error handling. Errors can be
/// classified at three levels:
///
/// * status_code: generic kind of error. We re-use HTTP status codes since
///   they are well known.
/// * title: a specific kind of error. For example, "Table not found."
/// * details: the details for an instance of the error.
pub struct InnerError {
    /// A short description of the error. For example, "Table not found.".
    ///
    /// This should not contain details of the instance of the error. Instead,
    /// save that for `details`. This should be the same for all instances of
    /// the same error, and thus is a static string.
    pub title: &'static str,
    /// A longer description of the error, including details specific to the
    /// instance. For example, "Table 'foo' not found in database 'bar'.".
    pub details: String,
    /// An HTTP status code that best describes the error.
    pub status_code: StatusCode,
    /// The underlying cause of the error, if any.
    pub cause: Option<BoxedError>,
    /// The backtrace or location of the error.
    pub backtrace: MaybeBacktrace,
}

// TODO: Include version of library
// TODO: include optional github URL format
pub enum MaybeBacktrace {
    Captured(Backtrace),
    Location(&'static std::panic::Location<'static>),
    None,
}

// /// Allocates error on the heap and then places `e` into it.
// #[inline]
// pub fn box_error(e: impl std::error::Error + Send + Sync + 'static) -> BoxedError {
//     Box::new(e)
// }

// #[derive(Debug, Snafu)]
// #[snafu(visibility(pub))]
// pub enum Error {
//     #[snafu(display("Invalid user input: {source}, {location}"))]
//     InvalidInput {
//         source: BoxedError,
//         location: Location,
//     },
//     #[snafu(display("Dataset already exists: {uri}, {location}"))]
//     DatasetAlreadyExists { uri: String, location: Location },
//     #[snafu(display("Append with different schema: {difference}, location: {location}"))]
//     SchemaMismatch {
//         difference: String,
//         location: Location,
//     },
//     #[snafu(display("Dataset at path {path} was not found: {source}, {location}"))]
//     DatasetNotFound {
//         path: String,
//         source: BoxedError,
//         location: Location,
//     },
//     #[snafu(display("Encountered corrupt file {path}: {source}, {location}"))]
//     CorruptFile {
//         path: object_store::path::Path,
//         source: BoxedError,
//         location: Location,
//         // TODO: add backtrace?
//     },
//     #[snafu(display("Not supported: {source}, {location}"))]
//     NotSupported {
//         source: BoxedError,
//         location: Location,
//     },
//     #[snafu(display("Commit conflict for version {version}: {source}, {location}"))]
//     CommitConflict {
//         version: u64,
//         source: BoxedError,
//         location: Location,
//     },
//     #[snafu(display("Encountered internal error. Please file a bug report at https://github.com/lancedb/lance/issues. {message}, {location}"))]
//     Internal { message: String, location: Location },
//     #[snafu(display("A prerequisite task failed: {message}, {location}"))]
//     PrerequisiteFailed { message: String, location: Location },
//     #[snafu(display("LanceError(Arrow): {message}, {location}"))]
//     Arrow { message: String, location: Location },
//     #[snafu(display("LanceError(Schema): {message}, {location}"))]
//     Schema { message: String, location: Location },
//     #[snafu(display("Not found: {uri}, {location}"))]
//     NotFound { uri: String, location: Location },
//     #[snafu(display("LanceError(IO): {source}, {location}"))]
//     IO {
//         source: BoxedError,
//         location: Location,
//     },
//     #[snafu(display("LanceError(Index): {message}, {location}"))]
//     Index { message: String, location: Location },
//     #[snafu(display("Lance index not found: {identity}, {location}"))]
//     IndexNotFound {
//         identity: String,
//         location: Location,
//     },
//     #[snafu(display("Cannot infer storage location from: {message}"))]
//     InvalidTableLocation { message: String },
//     /// Stream early stop
//     Stop,
//     #[snafu(display("Wrapped error: {error}, {location}"))]
//     Wrapped {
//         error: BoxedError,
//         location: Location,
//     },
//     #[snafu(display("Cloned error: {message}, {location}"))]
//     Cloned { message: String, location: Location },
//     #[snafu(display("Query Execution error: {message}, {location}"))]
//     Execution { message: String, location: Location },
//     #[snafu(display("Ref is invalid: {message}"))]
//     InvalidRef { message: String },
//     #[snafu(display("Ref conflict error: {message}"))]
//     RefConflict { message: String },
//     #[snafu(display("Ref not found error: {message}"))]
//     RefNotFound { message: String },
//     #[snafu(display("Cleanup error: {message}"))]
//     Cleanup { message: String },
//     #[snafu(display("Version not found error: {message}"))]
//     VersionNotFound { message: String },
//     #[snafu(display("Version conflict error: {message}"))]
//     VersionConflict {
//         message: String,
//         major_version: u16,
//         minor_version: u16,
//         location: Location,
//     },
// }

pub type Result<T> = std::result::Result<T, Error>;
pub type ArrowResult<T> = std::result::Result<T, ArrowError>;
#[cfg(feature = "datafusion")]
pub type DataFusionResult<T> = std::result::Result<T, datafusion_common::DataFusionError>;

impl From<ArrowError> for Error {
    #[track_caller]
    fn from(e: ArrowError) -> Self {
        let e = if let ArrowError::ExternalError(err) = e {
            match err.downcast::<Error>() {
                Ok(err) => return *err,
                Err(err) => ArrowError::ExternalError(err),
            }
        } else {
            e
        };

        let err = match &e {
            ArrowError::ParseError(msg) => {
                Self::new(StatusCode::BAD_REQUEST, "Parsing error", msg.to_string())
            }
            ArrowError::CastError(msg) => Self::new(StatusCode::BAD_REQUEST, "Casting error", msg),
            ArrowError::ArithmeticOverflow(msg) => {
                Self::new(StatusCode::BAD_REQUEST, "Arithmetic overflow", msg)
            }
            ArrowError::DivideByZero => {
                Self::new(StatusCode::BAD_REQUEST, "Divide by zero", "Divide by zero")
            }
            ArrowError::ExternalError(_) => {
                Self::new(StatusCode::INTERNAL_SERVER_ERROR, "External error", "")
            }
            _ => Self::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Arrow error",
                e.to_string(),
            ),
        };
        err.with_cause(e)
    }
}

impl From<std::io::Error> for Error {
    #[track_caller]
    fn from(e: std::io::Error) -> Self {
        use std::io::ErrorKind::*;
        let err = match e.kind() {
            NotFound => Self::new(StatusCode::NOT_FOUND, "Not found", e.to_string()),
            PermissionDenied => {
                Self::new(StatusCode::FORBIDDEN, "Permission denied", e.to_string())
            }
            _ => Self::new(StatusCode::INTERNAL_SERVER_ERROR, "IO error", e.to_string()),
        };
        err.with_cause(e)
    }
}

impl From<object_store::Error> for Error {
    #[track_caller]
    fn from(e: object_store::Error) -> Self {
        let err = match &e {
            object_store::Error::NotFound { .. } => {
                Self::new(StatusCode::NOT_FOUND, "Not found", e.to_string())
            }
            _ => Self::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Object store error",
                e.to_string(),
            ),
        };
        err.with_cause(e)
    }
}

impl From<prost::DecodeError> for Error {
    #[track_caller]
    fn from(e: prost::DecodeError) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Protobuf decode error",
            e.to_string(),
        )
        .with_cause(e)
        .with_backtrace()
    }
}

impl From<prost::EncodeError> for Error {
    #[track_caller]
    fn from(e: prost::EncodeError) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Protobuf encode error",
            e.to_string(),
        )
        .with_cause(e)
        .with_backtrace()
    }
}

impl From<prost::UnknownEnumValue> for Error {
    #[track_caller]
    fn from(e: prost::UnknownEnumValue) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Protobuf unknown enum value",
            e.to_string(),
        )
        .with_cause(e)
        .with_backtrace()
    }
}

impl From<tokio::task::JoinError> for Error {
    #[track_caller]
    fn from(e: tokio::task::JoinError) -> Self {
        Self::new(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Tokio join error",
            e.to_string(),
        )
        .with_cause(e)
    }
}

impl From<object_store::path::Error> for Error {
    #[track_caller]
    fn from(e: object_store::path::Error) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "Path error", e.to_string()).with_cause(e)
    }
}

impl From<url::ParseError> for Error {
    #[track_caller]
    fn from(e: url::ParseError) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "URL parse error", e.to_string()).with_cause(e)
    }
}

impl From<serde_json::Error> for Error {
    #[track_caller]
    fn from(e: serde_json::Error) -> Self {
        use serde_json::error::Category;
        match e.classify() {
            Category::Io => std::io::Error::from(e).into(),
            Category::Data | Category::Syntax | Category::Eof => {
                Self::new(StatusCode::BAD_REQUEST, "JSON parse error", e.to_string()).with_cause(e)
            }
        }
    }
}

impl From<Error> for ArrowError {
    fn from(value: Error) -> Self {
        Self::ExternalError(Box::new(value))
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_sql::sqlparser::parser::ParserError> for Error {
    #[track_caller]
    fn from(e: datafusion_sql::sqlparser::parser::ParserError) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "SQL parse error", e.to_string()).with_cause(e)
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_sql::sqlparser::tokenizer::TokenizerError> for Error {
    #[track_caller]
    fn from(e: datafusion_sql::sqlparser::tokenizer::TokenizerError) -> Self {
        Self::new(StatusCode::BAD_REQUEST, "SQL tokenize error", e.to_string()).with_cause(e)
    }
}

#[cfg(feature = "datafusion")]
impl From<Error> for datafusion_common::DataFusionError {
    #[track_caller]
    fn from(e: Error) -> Self {
        Self::External(Box::new(e))
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_common::DataFusionError> for Error {
    #[track_caller]
    fn from(e: datafusion_common::DataFusionError) -> Self {
        use datafusion_common::DataFusionError::*;
        // Early returns
        let e = match e {
            External(err) => match err.downcast::<Error>() {
                Ok(err) => return *err,
                Err(err) => External(err),
            },
            ArrowError(err, _) => return err.into(),
            ObjectStore(err) => return err.into(),
            IoError(err) => return err.into(),
            SQL(err, _) => return err.into(),
            Context(_, err) => return (*err).into(),
            _ => e,
        };

        let err = match &e {
            SchemaError(err, _) => {
                Self::new(StatusCode::BAD_REQUEST, "Schema error", err.to_string())
            }
            _ => Self::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "DataFusion error",
                e.to_string(),
            ),
        };
        err.with_cause(e)
    }
}

// This is a bit odd but some object_store functions only accept
// Stream<Result<T, ObjectStoreError>> and so we need to convert
// to ObjectStoreError to call the methods.
impl From<Error> for object_store::Error {
    fn from(err: Error) -> Self {
        Self::Generic {
            store: "lance",
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
/// only have Error::Cloned with the status, title, and details. The cause and
/// backtrace will be lost.
pub struct CloneableError(pub Error);

impl Clone for CloneableError {
    #[track_caller]
    fn clone(&self) -> Self {
        Self(Error::new(
            self.0.status_code,
            self.0.title,
            self.0.details.clone(),
        ))
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
        // TODO: fix this.
        // match f().unwrap_err() {
        //     Error { location, .. } => {
        //         // +4 is the beginning of object_store::Error::Generic...
        //         assert_eq!(location.line, current_fn.line() + 4, "{}", location)
        //     }
        //     #[allow(unreachable_patterns)]
        //     _ => panic!("expected ObjectStore error"),
        // }
    }
}
