use super::*;

#[derive(Debug)]
pub enum DatabaseError {
    Io(std::io::Error),
    Libsql(libsql::Error),
    MissingHomeDirectory,
    InvalidTimestamp(String),
    InvalidEmbeddingLength(usize),
    InvalidEmbedding(String),
    InvalidEnumValue(String),
}

impl std::fmt::Display for DatabaseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(f, "io error: {error}"),
            Self::Libsql(error) => write!(f, "database error: {error}"),
            Self::MissingHomeDirectory => write!(f, "HOME is not set"),
            Self::InvalidTimestamp(value) => write!(f, "invalid timestamp: {value}"),
            Self::InvalidEmbeddingLength(length) => {
                write!(f, "invalid embedding blob length: {length}")
            }
            Self::InvalidEmbedding(message) => write!(f, "invalid embedding: {message}"),
            Self::InvalidEnumValue(message) => write!(f, "invalid stored value: {message}"),
        }
    }
}

impl std::error::Error for DatabaseError {}

impl From<std::io::Error> for DatabaseError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<libsql::Error> for DatabaseError {
    fn from(value: libsql::Error) -> Self {
        Self::Libsql(value)
    }
}

impl From<DatabaseError> for TruesightError {
    fn from(value: DatabaseError) -> Self {
        TruesightError::Database(value.to_string())
    }
}
