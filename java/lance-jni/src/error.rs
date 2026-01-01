// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::str::Utf8Error;

use arrow_schema::ArrowError;
use jni::{errors::Error as JniError, JNIEnv};
use lance::Error as LanceError;
use lance_namespace::error::NamespaceError;
use serde_json::Error as JsonError;

#[derive(Debug, PartialEq, Eq)]
pub enum JavaExceptionClass {
    IllegalArgumentException,
    IOException,
    RuntimeException,
    UnsupportedOperationException,
    AlreadyInException,
    LanceNamespaceException,
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
            Self::LanceNamespaceException => "org/lance/namespace/errors/LanceNamespaceException",
        }
    }
}

#[derive(Debug)]
pub struct Error {
    message: String,
    java_class: JavaExceptionClass,
    namespace_error_code: Option<u32>,
}

impl Error {
    pub fn new(message: String, java_class: JavaExceptionClass) -> Self {
        Self {
            message,
            java_class,
            namespace_error_code: None,
        }
    }

    pub fn runtime_error(message: String) -> Self {
        Self {
            message,
            java_class: JavaExceptionClass::RuntimeException,
            namespace_error_code: None,
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

    pub fn namespace_error(code: u32, message: String) -> Self {
        Self {
            message,
            java_class: JavaExceptionClass::LanceNamespaceException,
            namespace_error_code: Some(code),
        }
    }

    pub fn in_exception() -> Self {
        Self {
            message: String::default(),
            java_class: JavaExceptionClass::AlreadyInException,
            namespace_error_code: None,
        }
    }

    pub fn throw(&self, env: &mut JNIEnv) {
        if self.java_class == JavaExceptionClass::AlreadyInException {
            // An exception is already in progress, so we don't need to throw another one.
            return;
        }

        // For namespace errors, throw the specific LanceNamespaceException
        if self.java_class == JavaExceptionClass::LanceNamespaceException {
            if let Some(code) = self.namespace_error_code {
                // Call LanceNamespaceException.fromCode static method
                if self.throw_namespace_exception(env, code).is_err() {
                    // lance-namespace is bundled as a dependency, so the exception classes
                    // should always be available. Panic if they're not.
                    panic!(
                        "Failed to throw LanceNamespaceException (code={}). \
                        org.lance.namespace.errors.LanceNamespaceException and ErrorCode classes \
                        must be available in the classpath.",
                        code
                    );
                }
                return;
            }
        }

        if let Err(e) = env.throw_new(self.java_class.as_str(), &self.message) {
            eprintln!("Error when throwing Java exception: {:?}", e.to_string());
            panic!("Error when throwing Java exception: {:?}", e);
        }
    }

    fn throw_namespace_exception(
        &self,
        env: &mut JNIEnv,
        code: u32,
    ) -> std::result::Result<(), ()> {
        // Try to find and call the LanceNamespaceException constructor
        // that takes ErrorCode and message
        let class_name = "org/lance/namespace/errors/LanceNamespaceException";
        let error_code_class = "org/lance/namespace/errors/ErrorCode";

        // Find the ErrorCode.fromCode method
        let error_code_cls = env.find_class(error_code_class).map_err(|_| ())?;
        let from_code_method = env
            .get_static_method_id(
                &error_code_cls,
                "fromCode",
                "(I)Lorg/lance/namespace/errors/ErrorCode;",
            )
            .map_err(|_| ())?;
        let error_code_obj = unsafe {
            env.call_static_method_unchecked(
                &error_code_cls,
                from_code_method,
                jni::signature::ReturnType::Object,
                &[jni::sys::jvalue {
                    i: code as jni::sys::jint,
                }],
            )
        }
        .map_err(|_| ())?;

        let error_code = match error_code_obj {
            jni::objects::JValueGen::Object(obj) => obj,
            _ => return Err(()),
        };

        // Find the LanceNamespaceException class
        let exception_cls = env.find_class(class_name).map_err(|_| ())?;

        // Create message JString
        let message_str = env.new_string(&self.message).map_err(|_| ())?;

        // Find constructor (ErrorCode, String)
        let constructor = env
            .get_method_id(
                &exception_cls,
                "<init>",
                "(Lorg/lance/namespace/errors/ErrorCode;Ljava/lang/String;)V",
            )
            .map_err(|_| ())?;

        // Create the exception object
        let exception_obj = unsafe {
            env.new_object_unchecked(
                &exception_cls,
                constructor,
                &[
                    jni::sys::jvalue {
                        l: error_code.as_raw(),
                    },
                    jni::sys::jvalue {
                        l: message_str.as_raw(),
                    },
                ],
            )
        }
        .map_err(|_| ())?;

        // Throw the exception
        env.throw(jni::objects::JThrowable::from(exception_obj))
            .map_err(|_| ())?;

        Ok(())
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
        match &err {
            LanceError::DatasetNotFound { .. }
            | LanceError::DatasetAlreadyExists { .. }
            | LanceError::CommitConflict { .. }
            | LanceError::InvalidInput { .. } => Self::input_error(err.to_string()),
            LanceError::IO { .. } => Self::io_error(err.to_string()),
            LanceError::NotSupported { .. } => Self::unsupported_error(err.to_string()),
            LanceError::NotFound { .. } => Self::io_error(err.to_string()),
            LanceError::Namespace { source, .. } => {
                // Try to downcast to NamespaceError and get the error code
                if let Some(ns_err) = source.downcast_ref::<NamespaceError>() {
                    Self::namespace_error(ns_err.code().as_u32(), ns_err.to_string())
                } else {
                    log::warn!(
                        "Failed to downcast NamespaceError source, falling back to runtime error. \
                         This may indicate a version mismatch. Source type: {:?}",
                        source
                    );
                    Self::runtime_error(err.to_string())
                }
            }
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
