// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_schema::ArrowError;
#[cfg(feature = "datafusion")]
use datafusion_common::DataFusionError;
use snafu::{Location, Snafu};
use std::backtrace::Backtrace;
use std::fmt;

use std::path::PathBuf;

type BoxedError = Box<dyn std::error::Error + Send + Sync + 'static>;
/// Allocates error on the heap and then places `e` into it.
#[inline]
pub fn box_error(e: impl std::error::Error + Send + Sync + 'static) -> BoxedError {
    Box::new(e)
}

/// All error variants that can be produced by Lance.
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

    #[snafu(display("Encountered corrupt file {}: {source}, {location}", path.display()))]
    CorruptFile {
        path: PathBuf,
        source: BoxedError,
        location: Location,
    },

    #[snafu(display("Has corrupt file: {message}, {location}"))]
    HasCorruptFile { message: String, location: Location },

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

    #[snafu(display("Retryable commit conflict for version {version}: {source}, {location}"))]
    RetryableCommitConflict {
        version: u64,
        source: BoxedError,
        location: Location,
    },

    #[snafu(display("Too many concurrent writers. {message}, {location}"))]
    TooMuchWriteContention { message: String, location: Location },

    #[snafu(display("Encountered internal error. Please file a bug report at https://github.com/lance-format/lance/issues. {message}, {location}"))]
    Internal { message: String, location: Location },

    #[snafu(display("A prerequisite task failed: {message}, {location}"))]
    PrerequisiteFailed { message: String, location: Location },

    #[snafu(display("Unprocessable: {message}, {location}"))]
    Unprocessable { message: String, location: Location },

    #[snafu(display("LanceError(Arrow): {message}, {location}"))]
    Arrow { message: String, location: Location },

    #[snafu(display("Schema error: {message}, {location}"))]
    Schema { message: String, location: Location },

    #[snafu(display("Not found: {uri}, {location}"))]
    NotFound { uri: String, location: Location },

    #[snafu(display("LanceError(IO): {source}, {location}"))]
    IO {
        source: BoxedError,
        location: Location,
    },

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

    #[snafu(display("Query Execution error: {message}, {location}"))]
    Execution { message: String, location: Location },

    #[snafu(display("Ref is invalid: {message}"))]
    InvalidRef { message: String },

    #[snafu(display("Ref conflict error: {message}"))]
    RefConflict { message: String },

    #[snafu(display("Ref not found error: {message}"))]
    RefNotFound { message: String },

    #[snafu(display("Cleanup error: {message}"))]
    Cleanup { message: String },

    #[snafu(display("Version not found error: {message}"))]
    VersionNotFound { message: String },

    #[snafu(display("Version conflict error: {message}"))]
    VersionConflict {
        message: String,
        major_version: u16,
        minor_version: u16,
        location: Location,
    },

    #[snafu(display("Namespace error: {source}, {location}"))]
    Namespace {
        source: BoxedError,
        location: Location,
    },
}

impl Error {
    /// Create a new error that includes a backtrace for internal errors
    pub fn new_internal_error(message: impl Into<String>, source: Option<BoxedError>) -> Self {
        let message = message.into();
        let location = std::panic::Location::caller().to_snafu_location();
        if let Some(source) = source {
            Self::Wrapped {
                error: source,
                location,
            }
        } else {
            Self::Internal { message, location }
        }
    }

    pub fn corrupt_file(
        path: impl ToString,
        message: impl Into<String>,
        location: Location,
    ) -> Self {
        let message = message.into();
        // Convert to PathBuf from string representation, handling both path types
        let path_buf = std::path::Path::new(&path.to_string())
            .to_str()
            .map(std::path::PathBuf::from)
            .unwrap_or_default();
        Self::CorruptFile {
            path: path_buf,
            source: message.into(),
            location,
        }
    }

    pub fn invalid_input(message: impl Into<String>, location: Location) -> Self {
        let message = message.into();
        Self::InvalidInput {
            source: message.into(),
            location,
        }
    }

    pub fn io(message: impl Into<String>, location: Location) -> Self {
        let message = message.into();
        Self::IO {
            source: message.into(),
            location,
        }
    }

    pub fn version_conflict(
        message: impl Into<String>,
        major_version: u16,
        minor_version: u16,
        location: Location,
    ) -> Self {
        let message = message.into();
        Self::VersionConflict {
            message,
            major_version,
            minor_version,
            location,
        }
    }
}

// ========== New Lance Error Type Implementation ==========

/// HTTP-like status codes for error classification
pub type StatusCode = u16;

/// Lance library version
pub static LANCE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Backtrace information for errors
#[derive(Debug)]
pub enum MaybeBacktrace {
    Captured(Backtrace),
    Location(&'static std::panic::Location<'static>),
    None,
}

/// Lance's generic error inner structure
#[derive(Debug)]
pub struct InnerError {
    /// HTTP status code-like error type
    pub status_code: StatusCode,
    /// A short static description of the error
    pub title: &'static str,
    /// A longer description with specific details
    pub details: String,
    /// The underlying cause of the error
    pub cause: Option<BoxedError>,
    /// Backtrace or location information
    pub backtrace: MaybeBacktrace,
    /// Lance version
    pub version: &'static str,
}

/// Lance's new improved error type
///
/// This error type follows the three-tier system:
/// 1. Status code (HTTP-inspired)
/// 2. Title (short static description)
/// 3. Details (dynamic specific information)
///
/// It also includes:
/// - Lance version
/// - Optional backtrace/location
/// - Optional cause
#[derive(Debug)]
pub struct LanceError(pub Box<InnerError>);

impl fmt::Display for LanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{title} ({status_code}): {details}",
            title = self.0.title,
            status_code = self.0.status_code,
            details = self.0.details
        )?;

        if let Some(ref cause) = self.0.cause {
            write!(f, "\nCaused by: {}", cause)?;
        }

        match &self.0.backtrace {
            MaybeBacktrace::Captured(backtrace) => {
                write!(f, "\nBacktrace:\n{}", backtrace)?;
            }
            MaybeBacktrace::Location(location) => {
                write!(f, "\nLocation: {}", location)?;
            }
            MaybeBacktrace::None => {}
        }

        write!(f, "\nLance version: {}", self.0.version)
    }
}

impl std::error::Error for LanceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.cause.as_deref().map(|e| e as &dyn std::error::Error)
    }
}

/// Builder for creating new Lance errors
#[derive(Debug)]
pub struct LanceErrorBuilder {
    status_code: StatusCode,
    title: &'static str,
    details: String,
    cause: Option<BoxedError>,
    backtrace: MaybeBacktrace,
}

impl LanceErrorBuilder {
    /// Create a new error builder with the given status code and title
    pub fn new(status_code: StatusCode, title: &'static str) -> Self {
        Self {
            status_code,
            title,
            details: String::new(),
            cause: None,
            backtrace: MaybeBacktrace::None,
        }
    }

    /// Set the detailed message for the error
    pub fn details(mut self, details: impl Into<String>) -> Self {
        self.details = details.into();
        self
    }

    /// Set the cause of the error
    pub fn cause(mut self, cause: impl std::error::Error + Send + Sync + 'static) -> Self {
        self.cause = Some(Box::new(cause));
        self
    }

    /// Set the backtrace for the error
    pub fn backtrace(mut self, backtrace: Backtrace) -> Self {
        self.backtrace = MaybeBacktrace::Captured(backtrace);
        self
    }

    /// Set the location for the error
    pub fn location(mut self, location: &'static std::panic::Location<'static>) -> Self {
        self.backtrace = MaybeBacktrace::Location(location);
        self
    }

    /// Capture the current backtrace if RUST_BACKTRACE is enabled
    #[track_caller]
    pub fn capture_backtrace(mut self) -> Self {
        self.backtrace = MaybeBacktrace::Captured(Backtrace::capture());
        self
    }

    /// Build the error
    pub fn build(self) -> LanceError {
        LanceError(Box::new(InnerError {
            status_code: self.status_code,
            title: self.title,
            details: self.details,
            cause: self.cause,
            backtrace: self.backtrace,
            version: LANCE_VERSION,
        }))
    }
}

// Helper functions to create common error types
impl LanceError {
    /// Create a 400 Bad Request error
    #[track_caller]
    pub fn bad_request(title: &'static str) -> LanceErrorBuilder {
        LanceErrorBuilder::new(400, title).location(std::panic::Location::caller())
    }

    /// Create a 404 Not Found error
    #[track_caller]
    pub fn not_found(title: &'static str) -> LanceErrorBuilder {
        LanceErrorBuilder::new(404, title).location(std::panic::Location::caller())
    }

    /// Create a 409 Conflict error
    #[track_caller]
    pub fn conflict(title: &'static str) -> LanceErrorBuilder {
        LanceErrorBuilder::new(409, title).location(std::panic::Location::caller())
    }

    /// Create a 422 Unprocessable Entity error
    #[track_caller]
    pub fn unprocessable(title: &'static str) -> LanceErrorBuilder {
        LanceErrorBuilder::new(422, title).location(std::panic::Location::caller())
    }

    /// Create a 500 Internal Server Error
    #[track_caller]
    pub fn internal(title: &'static str) -> LanceErrorBuilder {
        LanceErrorBuilder::new(500, title)
            .location(std::panic::Location::caller())
            .capture_backtrace()
    }

    /// Create a 501 Not Implemented error
    #[track_caller]
    pub fn not_implemented(title: &'static str) -> LanceErrorBuilder {
        LanceErrorBuilder::new(501, title).location(std::panic::Location::caller())
    }

    /// Create a new error with the given status code and title
    pub fn create(status_code: StatusCode, title: &'static str) -> LanceErrorBuilder {
        LanceErrorBuilder::new(status_code, title)
    }

    /// Convert to the legacy Error type for backward compatibility
    pub fn into_legacy(self, location: Location) -> Error {
        match self.0.status_code {
            400 => Error::InvalidInput {
                source: self.to_string().into(),
                location,
            },
            404 => Error::NotFound {
                uri: self.0.details,
                location,
            },
            409 => Error::CommitConflict {
                version: 0,
                source: self.to_string().into(),
                location,
            },
            422 => Error::Unprocessable {
                message: self.0.details,
                location,
            },
            _ => Error::Internal {
                message: self.0.details,
                location,
            },
        }
    }
}

// LanceError change to DataFusionError
#[cfg(feature = "datafusion")]
impl From<LanceError> for DataFusionError {
    #[track_caller]
    fn from(e: LanceError) -> Self {
        // if have DataFusion error then return
        if let Some(ref cause) = e.0.cause {
            if cause.downcast_ref::<Self>().is_some() {
                return Self::External(Box::new(e));
            }
        }

        Self::External(Box::new(e))
    }
}

#[cfg(feature = "datafusion")]
impl From<DataFusionError> for LanceError {
    #[track_caller]
    fn from(e: DataFusionError) -> Self {
        // unwrapper lance error
        if let DataFusionError::External(boxed) = e {
            match boxed.downcast::<Self>() {
                Ok(lance_error) => {
                    // extract lance error
                    return *lance_error;
                }
                Err(boxed) => {
                    // remake DataFusionError
                    let e = DataFusionError::External(boxed);
                    let (status_code, title) = match &e {
                        DataFusionError::SchemaError(..)
                        | DataFusionError::Plan(..)
                        | DataFusionError::Execution(..)
                        | DataFusionError::NotImplemented(..)
                        | DataFusionError::Internal(..) => (400, "Query error"),
                        DataFusionError::ArrowError(..) => (500, "Arrow error"),
                        DataFusionError::ObjectStore(..) => (500, "Storage error"),
                        _ => (500, "Internal error"),
                    };

                    return Self(Box::new(InnerError {
                        status_code,
                        title,
                        details: e.to_string(),
                        cause: Some(Box::new(e)),
                        backtrace: MaybeBacktrace::None,
                        version: LANCE_VERSION,
                    }));
                }
            }
        }

        // handle other DataFusionError
        let (status_code, title) = match &e {
            DataFusionError::SchemaError(..)
            | DataFusionError::Plan(..)
            | DataFusionError::Execution(..)
            | DataFusionError::NotImplemented(..)
            | DataFusionError::Internal(..) => (400, "Query error"),
            DataFusionError::ArrowError(..) => (500, "Arrow error"),
            DataFusionError::ObjectStore(..) => (500, "Storage error"),
            _ => (500, "Internal error"),
        };

        Self(Box::new(InnerError {
            status_code,
            title,
            details: e.to_string(),
            cause: Some(Box::new(e)),
            backtrace: MaybeBacktrace::None,
            version: LANCE_VERSION,
        }))
    }
}

//  ArrowError change to LanceError
impl From<ArrowError> for LanceError {
    #[track_caller]
    fn from(e: ArrowError) -> Self {
        match &e {
            ArrowError::SchemaError(_)
            | ArrowError::ComputeError(_)
            | ArrowError::ParseError(_)
            | ArrowError::InvalidArgumentError(_) => Self::bad_request("Arrow error")
                .details(format!("LanceError(Arrow): {}", e))
                .cause(e)
                .build(),
            // 这些类型的错误更可能是内部错误
            _ => Self::internal("Arrow error")
                .details(format!("LanceError(Arrow): {}", e))
                .cause(e)
                .capture_backtrace()
                .build(),
        }
    }
}

// old error change to new error
impl From<Error> for LanceError {
    #[track_caller]
    fn from(e: Error) -> Self {
        // old error type mapping to new error status code
        let (status_code, title) = match &e {
            Error::InvalidInput { .. } | Error::Arrow { .. } | Error::SchemaMismatch { .. } => {
                (400, "Invalid input")
            }
            Error::NotFound { .. } => (404, "Not found"),
            Error::CommitConflict { .. } | Error::RetryableCommitConflict { .. } => {
                (409, "Conflict")
            }
            Error::HasCorruptFile { .. } | Error::IO { .. } | Error::DatasetNotFound { .. } => {
                (500, "Internal error")
            }
            _ => (500, "Unknown error"),
        };

        Self::create(status_code, title)
            .details(format!("Legacy error: {}", e))
            .cause(e)
            .capture_backtrace()
            .build()
    }
}

// new lance error change to old error
impl From<LanceError> for Error {
    #[track_caller]
    fn from(e: LanceError) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        match e.0.status_code {
            400 => Self::InvalidInput {
                source: e.to_string().into(),
                location,
            },
            404 => Self::NotFound {
                uri: e.0.details,
                location,
            },
            409 => Self::CommitConflict {
                version: 0,
                source: e.to_string().into(),
                location,
            },
            422 => Self::Unprocessable {
                message: e.0.details,
                location,
            },
            _ => Self::Internal {
                message: e.0.details,
                location,
            },
        }
    }
}

// other error change to new lanceError
impl From<std::io::Error> for LanceError {
    #[track_caller]
    fn from(e: std::io::Error) -> Self {
        Self::internal("IO error")
            .details(format!("IO error: {}", e))
            .cause(e)
            .capture_backtrace()
            .build()
    }
}

impl From<object_store::Error> for LanceError {
    #[track_caller]
    fn from(e: object_store::Error) -> Self {
        Self::internal("Storage error")
            .details(format!("Storage error: {}", e))
            .cause(e)
            .capture_backtrace()
            .build()
    }
}

// ========== Legacy Error Implementations ==========

impl From<ArrowError> for Error {
    #[track_caller]
    fn from(e: ArrowError) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::Arrow {
            message: e.to_string(),
            location,
        }
    }
}

impl From<&ArrowError> for Error {
    #[track_caller]
    fn from(e: &ArrowError) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::Arrow {
            message: e.to_string(),
            location,
        }
    }
}

// Convert Legacy Error to ArrowError
impl From<Error> for ArrowError {
    #[track_caller]
    fn from(e: Error) -> Self {
        match &e {
            Error::InvalidInput { .. } | Error::Arrow { .. } | Error::SchemaMismatch { .. } => {
                Self::InvalidArgumentError(e.to_string())
            }
            _ => Self::ExternalError(Box::new(e)),
        }
    }
}

impl From<std::io::Error> for Error {
    #[track_caller]
    fn from(e: std::io::Error) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::IO {
            source: Box::new(e),
            location,
        }
    }
}

impl From<object_store::Error> for Error {
    #[track_caller]
    fn from(e: object_store::Error) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::IO {
            source: Box::new(e),
            location,
        }
    }
}

impl From<prost::DecodeError> for Error {
    #[track_caller]
    fn from(e: prost::DecodeError) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::IO {
            source: box_error(e),
            location,
        }
    }
}

impl From<prost::EncodeError> for Error {
    #[track_caller]
    fn from(e: prost::EncodeError) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::IO {
            source: box_error(e),
            location,
        }
    }
}

impl From<prost::UnknownEnumValue> for Error {
    #[track_caller]
    fn from(e: prost::UnknownEnumValue) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::IO {
            source: box_error(e),
            location,
        }
    }
}

impl From<tokio::task::JoinError> for Error {
    #[track_caller]
    fn from(e: tokio::task::JoinError) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::IO {
            source: box_error(e),
            location,
        }
    }
}

impl From<object_store::path::Error> for Error {
    #[track_caller]
    fn from(e: object_store::path::Error) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::IO {
            source: box_error(e),
            location,
        }
    }
}

impl From<url::ParseError> for Error {
    #[track_caller]
    fn from(e: url::ParseError) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::IO {
            source: box_error(e),
            location,
        }
    }
}

impl From<serde_json::Error> for Error {
    #[track_caller]
    fn from(e: serde_json::Error) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::InvalidInput {
            source: box_error(e),
            location,
        }
    }
}

#[cfg(feature = "datafusion")]
impl From<DataFusionError> for Error {
    #[track_caller]
    fn from(e: DataFusionError) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self::Wrapped {
            error: box_error(e),
            location,
        }
    }
}

#[cfg(feature = "datafusion")]
impl From<Error> for DataFusionError {
    #[track_caller]
    fn from(e: Error) -> Self {
        Self::External(Box::new(e))
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

// ========== Utility Functions and Traits ==========

pub trait LanceOptionExt<T> {
    /// Unwraps an option, returning an internal error if the option is None.
    ///
    /// Can be used when an option is expected to have a value.
    fn expect_ok(self) -> Result<T>;
}

impl<T> LanceOptionExt<T> for Option<T> {
    #[track_caller]
    fn expect_ok(self) -> Result<T> {
        let location = std::panic::Location::caller().to_snafu_location();
        self.ok_or_else(|| Error::Internal {
            message: "Expected option to have value".to_string(),
            location,
        })
    }
}

pub trait ToSnafuLocation {
    fn to_snafu_location(&'static self) -> Location;
}

impl ToSnafuLocation for std::panic::Location<'static> {
    fn to_snafu_location(&'static self) -> Location {
        Location::new(self.file(), self.line(), self.column())
    }
}

// ========== Type Definitions ==========

pub type Result<T> = std::result::Result<T, Error>;
pub type ArrowResult<T> = std::result::Result<T, ArrowError>;
#[cfg(feature = "datafusion")]
pub type DataFusionResult<T> = std::result::Result<T, DataFusionError>;

/// Wrap an error in a new error type that implements Clone
///
/// This is useful when two threads/streams share a common fallible source
/// The base error will always have the full error.  Any cloned results will
/// only have Error::Cloned with the to_string of the base error.
pub struct CloneableError(pub Error);

impl Clone for CloneableError {
    #[track_caller]
    fn clone(&self) -> Self {
        let location = std::panic::Location::caller().to_snafu_location();
        Self(Error::Cloned {
            message: self.0.to_string(),
            location,
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

#[track_caller]
pub fn get_caller_location() -> &'static std::panic::Location<'static> {
    std::panic::Location::caller()
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
