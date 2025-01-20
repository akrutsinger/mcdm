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
pub mod criteriatype;
pub mod dmatrix_ext;
pub mod errors;
pub mod normalization;
pub mod ranking;
pub mod validation;
pub mod weighting;

pub use criteriatype::CriteriaType;
pub use dmatrix_ext::DMatrixExt;
pub use errors::ValidationError;
pub use validation::{MatrixValidate, VectorValidate};
