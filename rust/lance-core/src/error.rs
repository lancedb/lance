// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;

use arrow_schema::ArrowError;
use snafu::{Location, Snafu};

type BoxedError = Box<dyn std::error::Error + Send + Sync + 'static>;

/// Error for when a requested field is not found in a schema.
///
/// This error computes suggestions lazily (only when displayed) to avoid
/// computing Levenshtein distance when the error is created but never shown.
#[derive(Debug)]
pub struct FieldNotFoundError {
    pub field_name: String,
    pub candidates: Vec<String>,
}

impl fmt::Display for FieldNotFoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Field '{}' not found.", self.field_name)?;
        let suggestion =
            crate::levenshtein::find_best_suggestion(&self.field_name, &self.candidates);
        if let Some(suggestion) = suggestion {
            write!(f, " Did you mean '{}'?", suggestion)?;
        }
        write!(f, "\nAvailable fields: [")?;
        for (i, candidate) in self.candidates.iter().take(10).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "'{}'", candidate)?;
        }
        if self.candidates.len() > 10 {
            let remaining = self.candidates.len() - 10;
            write!(f, ", ... and {} more]", remaining)?;
        } else {
            write!(f, "]")?;
        }
        Ok(())
    }
}

impl std::error::Error for FieldNotFoundError {}

/// Allocates error on the heap and then places `e` into it.
#[inline]
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
    #[snafu(display("Incompatible transaction: {source}, {location}"))]
    IncompatibleTransaction {
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
    #[snafu(display("LanceError(Schema): {message}, {location}"))]
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
    /// External error passed through from user code.
    ///
    /// This variant preserves errors that users pass into Lance APIs (e.g., via streams
    /// with custom error types). The original error can be recovered using [`Error::into_external`]
    /// or inspected using [`Error::external_source`].
    #[snafu(transparent)]
    External { source: BoxedError },

    /// A requested field was not found in a schema.
    #[snafu(transparent)]
    FieldNotFound { source: FieldNotFoundError },
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

    pub fn io(message: impl Into<String>, location: Location) -> Self {
        let message: String = message.into();
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
        let message: String = message.into();
        Self::VersionConflict {
            message,
            major_version,
            minor_version,
            location,
        }
    }

    pub fn not_found(uri: impl Into<String>) -> Self {
        Self::NotFound {
            uri: uri.into(),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }

    pub fn schema(message: impl Into<String>, location: Location) -> Self {
        let message: String = message.into();
        Self::Schema { message, location }
    }

    pub fn not_supported(message: impl Into<String>, location: Location) -> Self {
        let message: String = message.into();
        Self::NotSupported {
            source: message.into(),
            location,
        }
    }

    /// Create an External error from a boxed error source.
    pub fn external(source: BoxedError) -> Self {
        Self::External { source }
    }

    /// Create a FieldNotFound error with the given field name and available candidates.
    pub fn field_not_found(field_name: impl Into<String>, candidates: Vec<String>) -> Self {
        Self::FieldNotFound {
            source: FieldNotFoundError {
                field_name: field_name.into(),
                candidates,
            },
        }
    }

    /// Returns a reference to the external error source if this is an `External` variant.
    ///
    /// This allows downcasting to recover the original error type.
    pub fn external_source(&self) -> Option<&BoxedError> {
        match self {
            Self::External { source } => Some(source),
            _ => None,
        }
    }

    /// Consumes the error and returns the external source if this is an `External` variant.
    ///
    /// Returns `Err(self)` if this is not an `External` variant, allowing for chained handling.
    pub fn into_external(self) -> std::result::Result<BoxedError, Self> {
        match self {
            Self::External { source } => Ok(source),
            other => Err(other),
        }
    }
}

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
    fn to_snafu_location(&'static self) -> snafu::Location;
}

impl ToSnafuLocation for std::panic::Location<'static> {
    fn to_snafu_location(&'static self) -> snafu::Location {
        snafu::Location::new(self.file(), self.line(), self.column())
    }
}

pub type Result<T> = std::result::Result<T, Error>;
pub type ArrowResult<T> = std::result::Result<T, ArrowError>;
#[cfg(feature = "datafusion")]
pub type DataFusionResult<T> = std::result::Result<T, datafusion_common::DataFusionError>;

impl From<ArrowError> for Error {
    #[track_caller]
    fn from(e: ArrowError) -> Self {
        match e {
            ArrowError::ExternalError(source) => {
                // Try to downcast to lance_core::Error first to recover the original
                match source.downcast::<Self>() {
                    Ok(lance_err) => *lance_err,
                    Err(source) => Self::External { source },
                }
            }
            other => Self::Arrow {
                message: other.to_string(),
                location: std::panic::Location::caller().to_snafu_location(),
            },
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
            source: box_error(e),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<object_store::Error> for Error {
    #[track_caller]
    fn from(e: object_store::Error) -> Self {
        Self::IO {
            source: box_error(e),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<prost::DecodeError> for Error {
    #[track_caller]
    fn from(e: prost::DecodeError) -> Self {
        Self::IO {
            source: box_error(e),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<prost::EncodeError> for Error {
    #[track_caller]
    fn from(e: prost::EncodeError) -> Self {
        Self::IO {
            source: box_error(e),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<prost::UnknownEnumValue> for Error {
    #[track_caller]
    fn from(e: prost::UnknownEnumValue) -> Self {
        Self::IO {
            source: box_error(e),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<tokio::task::JoinError> for Error {
    #[track_caller]
    fn from(e: tokio::task::JoinError) -> Self {
        Self::IO {
            source: box_error(e),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<object_store::path::Error> for Error {
    #[track_caller]
    fn from(e: object_store::path::Error) -> Self {
        Self::IO {
            source: box_error(e),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

impl From<url::ParseError> for Error {
    #[track_caller]
    fn from(e: url::ParseError) -> Self {
        Self::IO {
            source: box_error(e),
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

impl From<Error> for ArrowError {
    fn from(value: Error) -> Self {
        match value {
            // Pass through external errors directly
            Error::External { source } => Self::ExternalError(source),
            // Preserve schema errors with their specific type
            Error::Schema { message, .. } => Self::SchemaError(message),
            // Wrap all other lance errors so they can be recovered
            e => Self::ExternalError(Box::new(e)),
        }
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_sql::sqlparser::parser::ParserError> for Error {
    #[track_caller]
    fn from(e: datafusion_sql::sqlparser::parser::ParserError) -> Self {
        Self::IO {
            source: box_error(e),
            location: std::panic::Location::caller().to_snafu_location(),
        }
    }
}

#[cfg(feature = "datafusion")]
impl From<datafusion_sql::sqlparser::tokenizer::TokenizerError> for Error {
    #[track_caller]
    fn from(e: datafusion_sql::sqlparser::tokenizer::TokenizerError) -> Self {
        Self::IO {
            source: box_error(e),
            location: std::panic::Location::caller().to_snafu_location(),
        }
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
        let location = std::panic::Location::caller().to_snafu_location();
        match e {
            datafusion_common::DataFusionError::SQL(..)
            | datafusion_common::DataFusionError::Plan(..)
            | datafusion_common::DataFusionError::Configuration(..) => Self::InvalidInput {
                source: box_error(e),
                location,
            },
            datafusion_common::DataFusionError::SchemaError(..) => Self::Schema {
                message: e.to_string(),
                location,
            },
            datafusion_common::DataFusionError::ArrowError(arrow_err, _) => Self::from(*arrow_err),
            datafusion_common::DataFusionError::NotImplemented(..) => Self::NotSupported {
                source: box_error(e),
                location,
            },
            datafusion_common::DataFusionError::Execution(..) => Self::Execution {
                message: e.to_string(),
                location,
            },
            datafusion_common::DataFusionError::External(source) => {
                // Try to downcast to lance_core::Error first
                match source.downcast::<Self>() {
                    Ok(lance_err) => *lance_err,
                    Err(source) => Self::External { source },
                }
            }
            _ => Self::IO {
                source: box_error(e),
                location,
            },
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
    use std::fmt;

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

    #[derive(Debug)]
    struct MyCustomError {
        code: i32,
        message: String,
    }

    impl fmt::Display for MyCustomError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "MyCustomError({}): {}", self.code, self.message)
        }
    }

    impl std::error::Error for MyCustomError {}

    #[test]
    fn test_external_error_creation() {
        let custom_err = MyCustomError {
            code: 42,
            message: "test error".to_string(),
        };
        let err = Error::external(Box::new(custom_err));

        match &err {
            Error::External { source } => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 42);
                assert_eq!(recovered.message, "test error");
            }
            _ => panic!("Expected External variant"),
        }
    }

    #[test]
    fn test_external_source_method() {
        let custom_err = MyCustomError {
            code: 123,
            message: "source test".to_string(),
        };
        let err = Error::external(Box::new(custom_err));

        let source = err.external_source().expect("should have external source");
        let recovered = source.downcast_ref::<MyCustomError>().unwrap();
        assert_eq!(recovered.code, 123);

        // Test that non-External variants return None
        let io_err = Error::io("test", snafu::Location::new("test", 1, 1));
        assert!(io_err.external_source().is_none());
    }

    #[test]
    fn test_into_external_method() {
        let custom_err = MyCustomError {
            code: 456,
            message: "into test".to_string(),
        };
        let err = Error::external(Box::new(custom_err));

        match err.into_external() {
            Ok(source) => {
                let recovered = source.downcast::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 456);
            }
            Err(_) => panic!("Expected Ok"),
        }

        // Test that non-External variants return Err(self)
        let io_err = Error::io("test", snafu::Location::new("test", 1, 1));
        match io_err.into_external() {
            Err(Error::IO { .. }) => {}
            _ => panic!("Expected Err with IO variant"),
        }
    }

    #[test]
    fn test_arrow_external_error_conversion() {
        let custom_err = MyCustomError {
            code: 789,
            message: "arrow test".to_string(),
        };
        let arrow_err = ArrowError::ExternalError(Box::new(custom_err));
        let lance_err: Error = arrow_err.into();

        match lance_err {
            Error::External { source } => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 789);
            }
            _ => panic!("Expected External variant, got {:?}", lance_err),
        }
    }

    #[test]
    fn test_external_to_arrow_roundtrip() {
        let custom_err = MyCustomError {
            code: 999,
            message: "roundtrip".to_string(),
        };
        let lance_err = Error::external(Box::new(custom_err));
        let arrow_err: ArrowError = lance_err.into();

        match arrow_err {
            ArrowError::ExternalError(source) => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 999);
            }
            _ => panic!("Expected ExternalError variant"),
        }
    }

    #[cfg(feature = "datafusion")]
    #[test]
    fn test_datafusion_external_error_conversion() {
        let custom_err = MyCustomError {
            code: 111,
            message: "datafusion test".to_string(),
        };
        let df_err = datafusion_common::DataFusionError::External(Box::new(custom_err));
        let lance_err: Error = df_err.into();

        match lance_err {
            Error::External { source } => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 111);
            }
            _ => panic!("Expected External variant"),
        }
    }

    #[cfg(feature = "datafusion")]
    #[test]
    fn test_datafusion_arrow_external_error_conversion() {
        // Test the nested case: ArrowError::ExternalError inside DataFusionError::ArrowError
        let custom_err = MyCustomError {
            code: 222,
            message: "nested test".to_string(),
        };
        let arrow_err = ArrowError::ExternalError(Box::new(custom_err));
        let df_err = datafusion_common::DataFusionError::ArrowError(Box::new(arrow_err), None);
        let lance_err: Error = df_err.into();

        match lance_err {
            Error::External { source } => {
                let recovered = source.downcast_ref::<MyCustomError>().unwrap();
                assert_eq!(recovered.code, 222);
            }
            _ => panic!("Expected External variant, got {:?}", lance_err),
        }
    }

    /// Test that lance_core::Error round-trips through ArrowError.
    ///
    /// This simulates the case where a user defines an iterator in terms of
    /// lance_core::Error, and the error goes through Arrow's error type
    /// (e.g., via RecordBatchIterator) before being converted back.
    #[test]
    fn test_lance_error_roundtrip_through_arrow() {
        let original = Error::invalid_input(
            "test validation error",
            snafu::Location::new("test.rs", 10, 1),
        );

        // Simulate what happens when using ? in an Arrow context
        let arrow_err: ArrowError = original.into();

        // Convert back to lance error (as happens when Lance consumes the stream)
        let recovered: Error = arrow_err.into();

        // Should get back the original lance error directly (not wrapped in External)
        match recovered {
            Error::InvalidInput { .. } => {
                assert!(recovered.to_string().contains("test validation error"));
            }
            _ => panic!("Expected InvalidInput variant, got {:?}", recovered),
        }
    }

    /// Test that lance_core::Error round-trips through DataFusionError.
    ///
    /// This simulates the case where a user defines a stream in terms of
    /// lance_core::Error, and the error goes through DataFusion's error type
    /// (e.g., via SendableRecordBatchStream) before being converted back.
    #[cfg(feature = "datafusion")]
    #[test]
    fn test_lance_error_roundtrip_through_datafusion() {
        let original = Error::invalid_input(
            "test validation error",
            snafu::Location::new("test.rs", 10, 1),
        );

        // Simulate what happens when using ? in a DataFusion context
        let df_err: datafusion_common::DataFusionError = original.into();

        // Convert back to lance error (as happens when Lance consumes the stream)
        let recovered: Error = df_err.into();

        // Should get back the original lance error directly (not wrapped in External)
        match recovered {
            Error::InvalidInput { .. } => {
                assert!(recovered.to_string().contains("test validation error"));
            }
            _ => panic!("Expected InvalidInput variant, got {:?}", recovered),
        }
    }
}
