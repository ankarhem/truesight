//! A minimal fixture library for testing code extraction.

mod utils;

use crate::utils::format_name;

/// Represents a user in the system.
pub struct User {
    pub id: u64,
    pub name: String,
    pub email: String,
}

/// Possible errors when validating users.
pub enum ValidationError {
    InvalidEmail,
    NameTooShort,
    InvalidId,
}

/// Trait for entities that can be validated.
pub trait Validatable {
    fn validate(&self) -> Result<(), ValidationError>;
}

impl User {
    /// Creates a new user with the given details.
    pub fn new(id: u64, name: &str, email: &str) -> Self {
        Self {
            id,
            name: format_name(name),
            email: email.to_string(),
        }
    }

    /// Returns a formatted display string for the user.
    pub fn display_name(&self) -> String {
        format!("{} <{}>", self.name, self.email)
    }
}

impl Validatable for User {
    fn validate(&self) -> Result<(), ValidationError> {
        if !self.email.contains('@') {
            return Err(ValidationError::InvalidEmail);
        }
        if self.name.len() < 2 {
            return Err(ValidationError::NameTooShort);
        }
        if self.id == 0 {
            return Err(ValidationError::InvalidId);
        }
        Ok(())
    }
}

/// Validates an email address format.
pub fn validate_email(email: &str) -> bool {
    email.contains('@') && email.contains('.')
}

/// Calculates a simple checksum for data integrity.
pub fn calculate_checksum(data: &[u8]) -> u32 {
    data.iter().map(|&b| b as u32).sum()
}
