use std::io;

/// An error returned by vector database operations.
///
// Implementers: convert dependency errors at crate boundaries and write the message yourself;
// forwarding a dependency's error text would couple the public messages to dependency internals.
/// `Error` is non-exhaustive, so new categories can be added without breaking downstream matches.
/// Error messages are stable across dependency upgrades.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A caller supplied an invalid argument or document value.
    #[non_exhaustive]
    #[error("invalid argument: {message}")]
    InvalidArgument {
        /// An explanation of the invalid input.
        message: String,
    },
    /// A requested object does not exist.
    #[non_exhaustive]
    #[error("not found: {message}")]
    NotFound {
        /// A description of the missing object.
        message: String,
    },
    /// The object to be created already exists.
    #[non_exhaustive]
    #[error("already exists: {message}")]
    AlreadyExists {
        /// A description of the conflicting object.
        message: String,
    },
    /// Stored data violates a persistence or encoding invariant.
    #[non_exhaustive]
    #[error("corruption: {message}")]
    Corruption {
        /// A description of the corrupt data.
        message: String,
    },
    // std::io::Error may appear here because the standard library is a stable boundary; errors
    // from storage dependencies are converted before reaching this variant.
    /// An operating-system I/O operation failed.
    ///
    /// The underlying [`std::io::Error`] is available through
    /// [`source`](std::error::Error::source).
    #[non_exhaustive]
    #[error("I/O error: {source}")]
    Io {
        /// The original standard-library I/O error.
        #[source]
        source: io::Error,
    },
    /// The requested behavior is intentionally unsupported.
    #[non_exhaustive]
    #[error("unsupported: {message}")]
    Unsupported {
        /// A description of the unsupported behavior.
        message: String,
    },
    /// An internal invariant was violated, with no evidence of persisted-data corruption.
    #[non_exhaustive]
    #[error("internal error: {message}")]
    Internal {
        /// A description of the internal failure.
        message: String,
    },
}

impl Error {
    /// Creates an invalid-argument error from a message.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument {
            message: message.into(),
        }
    }

    /// Creates a not-found error from a message.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self::NotFound {
            message: message.into(),
        }
    }

    /// Creates an already-exists error from a message.
    pub fn already_exists(message: impl Into<String>) -> Self {
        Self::AlreadyExists {
            message: message.into(),
        }
    }

    /// Creates a corruption error from a message.
    pub fn corruption(message: impl Into<String>) -> Self {
        Self::Corruption {
            message: message.into(),
        }
    }

    /// Creates an I/O error that retains the standard-library error as its source.
    pub fn io(source: io::Error) -> Self {
        Self::Io { source }
    }

    /// Creates an unsupported-operation error from a message.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    /// Creates an internal error from a message.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Returns the error's message.
    ///
    /// I/O errors return `None`; their standard-library source is available through
    /// [`std::error::Error::source`]. Categories can be classified with non-exhaustive patterns such
    /// as `matches!(error, Error::InvalidArgument { .. })`.
    pub fn message(&self) -> Option<&str> {
        match self {
            Self::InvalidArgument { message }
            | Self::NotFound { message }
            | Self::AlreadyExists { message }
            | Self::Corruption { message }
            | Self::Unsupported { message }
            | Self::Internal { message } => Some(message),
            Self::Io { .. } => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(source: io::Error) -> Self {
        Self::io(source)
    }
}

/// A result whose error uses the vector database's stable error vocabulary.
pub type Result<T> = std::result::Result<T, Error>;
