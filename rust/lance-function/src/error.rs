// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fmt;

use serde::{Deserialize, Serialize};

/// Stable Function v1 failure categories, independent of transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The artifact or its requirements cannot be loaded by this worker.
    Incompatible,
    /// Module import, initialization data, or factory initialization failed.
    InitializationFailed,
    /// Applying the function or closing it raised an ordinary execution error.
    ExecutionFailed,
    /// Output violated the declared type, length, or nullability contract.
    InvalidOutput,
    /// A required execution resource or budget was exhausted.
    ResourceExhausted,
    /// The caller cancelled the attempt; subsequent output must be discarded.
    Cancelled,
}

impl ErrorCode {
    /// The wire spelling used by Function v1 control protocols.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Incompatible => "incompatible",
            Self::InitializationFailed => "initialization_failed",
            Self::ExecutionFailed => "execution_failed",
            Self::InvalidOutput => "invalid_output",
            Self::ResourceExhausted => "resource_exhausted",
            Self::Cancelled => "cancelled",
        }
    }
}

/// A stable error category and diagnostic message; no retry policy is implied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    /// Machine-readable classification.
    pub code: ErrorCode,
    /// Context sufficient to locate the rejected property or failed operation.
    pub message: String,
}

impl Error {
    /// Construct a failure from its transport-independent parts.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub(crate) fn incompatible(message: impl Into<String>) -> Self {
        Self::new(ErrorCode::Incompatible, message)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for Error {}

/// The result of artifact or interface validation.
pub type Result<T> = std::result::Result<T, Error>;
