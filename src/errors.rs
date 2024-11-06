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

#[derive(Debug)]
pub struct MissingElementError {
    details: String
}

impl MissingElementError {
    pub fn new(what: &str) -> MissingElementError {
        MissingElementError {
            details: "Analysis is missing some element: ".to_owned() + what
        }
    }
}

impl Display for MissingElementError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for MissingElementError {
    fn description(&self) -> &str {
        &self.details
    }
}

#[derive(Debug)]
pub struct InconsistencyError {
    details: String
}

impl InconsistencyError {
    pub fn new(what: &str) -> InconsistencyError {
        InconsistencyError {
            details: "Inconsistency in analysis: ".to_owned() + what
        }
    }
}

impl Display for InconsistencyError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for InconsistencyError {
    fn description(&self) -> &str {
        &self.details
    }
}