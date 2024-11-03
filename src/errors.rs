use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub struct DataLengthError {
    details: String
}

impl DataLengthError {
    pub fn new() -> DataLengthError {
        DataLengthError {
            details: "Length of data was not a multiple of 8 * columns".to_string()
        }
    }
}

impl Display for DataLengthError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for DataLengthError {
    fn description(&self) -> &str {
        &self.details
    }
}