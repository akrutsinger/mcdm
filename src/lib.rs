//! The `mcdm` crate offers a set of utilities for implementing Multiple-Criteria Decision Making
//! (MCDM) techniques in Rust, enabling users to analyze and rank alternatives based on multiple
//! conflicting criteria.
//!
//! # Example
//!
//! ```rust
//! use mcdm::{McdmError, Rank, Normalize, Weight, CriteriaTypes};
//! use nalgebra::dmatrix;
//!
//! fn main() -> Result<(), McdmError> {
//!     // Define the decision matrix (alternatives x criteria)
//!     let alternatives = dmatrix![4.0, 7.0, 8.0; 2.0, 9.0, 6.0; 3.0, 6.0, 9.0];
//!     let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1])?;
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
#![no_std]

#[cfg(feature = "std")]
extern crate std;

mod float;
use float::Float;

pub mod correlation;
pub mod criteria;
pub mod dmatrix_ext;
pub mod errors;
pub mod normalization;
pub mod ranking;
pub mod validation;
pub mod weighting;

pub use crate::correlation::*;
pub use crate::criteria::*;
pub use crate::dmatrix_ext::*;
pub use crate::errors::*;
pub use crate::normalization::*;
pub use crate::ranking::*;
pub use crate::validation::*;
pub use crate::weighting::*;
