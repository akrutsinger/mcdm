//! The `mcdm` crate offers a set of utilities for implementing Multiple-Criteria Decision Making
//! (MCDM) techniques in Rust, enabling users to analyze and rank alternatives based on multiple
//! conflicting criteria.
//!
//! # Example
//!
//! ```rust
//! use mcdm::{
//!     errors::McdmError,
//!     methods::{Rank, TOPSIS},
//!     normalization::{MinMax, Normalize},
//!     weights::{Equal, Weight},
//!     CriteriaType,
//! };
//! use ndarray::array;
//!
//! fn main() -> Result<(), McdmError> {
//!     // Define the decision matrix (alternatives x criteria)
//!     let alternatives = array![[4.0, 7.0, 8.0], [2.0, 9.0, 6.0], [3.0, 6.0, 9.0]];
//!     let criteria_types = CriteriaType::from_vec(vec![-1, 1, 1])?;
//!
//!     // Apply normalization using Min-Max
//!     let normalized_matrix = MinMax::normalize(&alternatives, &criteria_types)?;
//!
//!     // Alternatively, use equal weights
//!     let equal_weights = Equal::weight(&normalized_matrix)?;
//!
//!     // Apply the TOPSIS method for ranking
//!     let ranking = TOPSIS::rank(&normalized_matrix, &equal_weights)?;
//!
//!     // Output the ranking
//!     println!("Ranking: {:.3?}", ranking);
//!
//!     Ok(())
//! }
//! ```

use errors::ValidationError;

/// A set of errors that can occur in the `mcdm` crate.
pub mod errors;
/// Techniques for ranking alternatives.
pub mod methods;
/// Normalization techniques for normalizing a decision matrix.
pub mod normalization;
/// Weighting methods for assigning weights to criteria of a normalized matrix.
pub mod weights;

/// Enum to represent the type of each criterion: either Cost or Profit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CriteriaType {
    /// Criterion where lower values are preferred (minimization).
    Cost,
    /// Criterion where higher values are preferred (maximization).
    Profit,
}

/// Create a new `CriteriaTypes` from an array of `i8` values (-1 for Cost, 1 for Profit).
///
/// # Arguments
///
/// * `types_array` - A 1D array where -1 represents Cost and 1 represents Profit.
///
/// # Returns
///
/// * `Result<Self, ValidationError>` - An array of `CriteriaTypes` or an error for invalid values.
impl CriteriaType {
    pub fn from_vec(types_array: Vec<i8>) -> Result<Vec<CriteriaType>, ValidationError> {
        types_array
            .into_iter()
            .map(|value| match value {
                -1 => Ok(CriteriaType::Cost),
                1 => Ok(CriteriaType::Profit),
                _ => Err(ValidationError::InvalidValue),
            })
            .collect::<Result<Vec<CriteriaType>, ValidationError>>()
    }

    /// Generate a vector of `CriteriaType::Profit` of the given length `len`.
    pub fn profits(len: usize) -> Vec<CriteriaType> {
        vec![CriteriaType::Profit; len]
    }

    /// Generate a vector of `CriteriaType::Cost` of the given length `len`.
    pub fn costs(len: usize) -> Vec<CriteriaType> {
        vec![CriteriaType::Cost; len]
    }
}
