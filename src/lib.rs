//! The `mcdm` crate offers a set of utilities for implementing Multiple-Criteria Decision Making
//! (MCDM) techniques in Rust, enabling users to analyze and rank alternatives based on multiple
//! conflicting criteria.
//!
//! # Example
//!
//! ```rust
//! use mcdm::{
//!     errors::McdmError, ranking::Rank, normalization::Normalize, weighting::Weight, CriteriaType,
//! };
//! use nalgebra::dmatrix;
//!
//! fn main() -> Result<(), McdmError> {
//!     // Define the decision matrix (alternatives x criteria)
//!     let alternatives = dmatrix![4.0, 7.0, 8.0; 2.0, 9.0, 6.0; 3.0, 6.0, 9.0];
//!     let criteria_types = CriteriaType::from(vec![-1, 1, 1])?;
//!
//!     // Apply normalization using Min-Max
//!     let normalized_matrix = alternatives.normalize_min_max(&criteria_types)?;
//!
//!     // Alternatively, use equal weights
//!     let equal_weights = normalized_matrix.weight_equal()?;
//!
//!     // Apply the TOPSIS method for ranking
//!     let ranking = normalized_matrix.rank_topsis(&equal_weights)?;
//!
//!     // Output the ranking
//!     println!("Ranking: {:.3}", ranking);
//!     // Ranking:
//!     //   ┌      ┐
//!     //   │ 0.626 │
//!     //   │ 0.414 │
//!     //   │ 0.500 │
//!     //   └      ┘
//!
//!     Ok(())
//! }
//! ```
pub mod dmatrix_ext;
pub mod errors;
pub mod normalization;
pub mod ranking;
pub mod weighting;

pub use dmatrix_ext::DMatrixExt;
pub use errors::ValidationError;

/// Enum to represent the type of each criterion: either Cost or Profit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CriteriaType {
    /// Criterion where lower values are preferred (minimization).
    Cost,
    /// Criterion where higher values are preferred (maximization).
    Profit,
}

impl CriteriaType {
    /// Converts an iterator of `i8` values (-1 for `Cost`, 1 for `Profit`) into a vector of `CriteriaType`.
    ///
    /// # Arguments
    ///
    /// * `iter` - An iterator of `i8` values. Each value should be either `-1` (for `Cost`) or `1`
    ///   (for `Profit`).
    ///
    /// # Returns
    ///
    /// * `Result<Vec<CriteriaType>, ValidationError>` - A vector of `CriteriaType` if the values
    ///   are valid, or an error if an invalid value is encountered.
    ///
    /// # Errors
    ///
    /// * `ValidationError::InvalidValue` - If the iterator contains values other than `-1` or `1`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mcdm::errors::ValidationError;
    /// use mcdm::CriteriaType;
    /// use nalgebra::dmatrix;
    ///
    /// let criteria_types = vec![-1, 1, 1]; // -1 for Cost, 1 for Profit
    /// let criteria = CriteriaType::from(criteria_types).unwrap();
    /// assert_eq!(criteria, vec![CriteriaType::Cost, CriteriaType::Profit, CriteriaType::Profit]);
    /// ```
    pub fn from<I>(iter: I) -> Result<Vec<CriteriaType>, ValidationError>
    where
        I: IntoIterator<Item = i8>,
    {
        iter.into_iter()
            .map(|value| match value {
                -1 => Ok(CriteriaType::Cost),
                1 => Ok(CriteriaType::Profit),
                _ => Err(ValidationError::InvalidValue),
            })
            .collect::<Result<Vec<CriteriaType>, ValidationError>>()
    }

    /// Generate a vector of [`CriteriaType::Profit`] of the given length `len`.
    pub fn profits(len: usize) -> Vec<CriteriaType> {
        vec![CriteriaType::Profit; len]
    }

    /// Generate a vector of [`CriteriaType::Cost`] of the given length `len`.
    pub fn costs(len: usize) -> Vec<CriteriaType> {
        vec![CriteriaType::Cost; len]
    }

    /// Switches each `Cost` to a `Profit` and each `Profit` to a `Cost` in the given vector.
    ///
    /// # Arguments
    ///
    /// * `types` - A vector of `CriteriaType`.
    ///
    /// # Returns
    ///
    /// * `Vec<CriteriaType>` - A vector with each `Cost` switched to a `Profit` and each `Profit`
    ///   switched to a `Cost`.
    pub fn switch(types: Vec<CriteriaType>) -> Vec<CriteriaType> {
        types
            .into_iter()
            .map(|t| match t {
                CriteriaType::Cost => CriteriaType::Profit,
                CriteriaType::Profit => CriteriaType::Cost,
            })
            .collect()
    }
}
