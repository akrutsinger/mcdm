//! Module for representing the type of each criterion: either Cost or Profit.

use crate::errors::ValidationError;
use nalgebra::DVector;
use std::convert::TryFrom;
use std::ops::{Deref, DerefMut, Index, IndexMut};

/// Enum to represent the type of each criterion: either Cost or Profit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CriterionType {
    /// Criterion where lower values are preferred (minimization).
    Cost,
    /// Criterion where higher values are preferred (maximization).
    Profit,
}

/// Alias for a vector of `CriterionType`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CriteriaTypes(DVector<CriterionType>);

impl TryFrom<i8> for CriterionType {
    type Error = ValidationError;

    fn try_from(value: i8) -> Result<Self, Self::Error> {
        match value {
            -1 => Ok(CriterionType::Cost),
            1 => Ok(CriterionType::Profit),
            _ => Err(ValidationError::InvalidValue),
        }
    }
}

impl CriteriaTypes {
    /// Create a new `CriteriaTypes` vector with specific values
    pub fn new(values: &[CriterionType]) -> Self {
        CriteriaTypes(DVector::from_vec(values.to_vec()))
    }

    /// Creates a new `CriteriaTypes` from a slice of `i8` values (-1 for `Cost`, 1 for `Profit`).
    ///
    /// # Arguments
    ///
    /// * `slice` - A slice of `i8` values. Each value should be either `-1` (for `Cost`) or `1`
    ///   (for `Profit`).
    ///
    /// # Returns
    ///
    /// * `Result<CriteriaTypes, ValidationError>` - A `CriteriaTypes` if the values
    ///   are valid, or an error if an invalid value is encountered.
    ///
    /// # Errors
    ///
    /// * [`ValidationError`](crate::errors::ValidationError) - If the slice contains values other
    ///   than `-1` or `1`.
    pub fn from_slice(values: &[i8]) -> Result<Self, ValidationError> {
        let mut criteria = DVector::from_element(values.len(), CriterionType::Cost);
        for (i, &v) in values.iter().enumerate() {
            criteria[i] = CriterionType::try_from(v)?;
        }
        Ok(CriteriaTypes(criteria))
    }

    /// Get the length of the criteria vector.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if the criteria vector is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Generate a vector of [`CriterionType::Profit`] of the given length `len`.
    pub fn all_profits(len: usize) -> Self {
        CriteriaTypes(DVector::from_element(len, CriterionType::Profit))
    }

    /// Generate a vector of [`CriterionType::Cost`] of the given length `len`.
    pub fn all_costs(len: usize) -> Self {
        CriteriaTypes(DVector::from_element(len, CriterionType::Cost))
    }

    /// Switches each `Cost` to a `Profit` and each `Profit` to a `Cost` in the given vector.
    ///
    /// # Arguments
    ///
    /// * `types` - A vector of `CriterionType`.
    ///
    /// # Returns
    ///
    /// * `CriteriaTypes` - A 1d vector with each `Cost` switched to a `Profit` and each `Profit`
    ///   switched to a `Cost`.
    pub fn switch(types: &CriteriaTypes) -> Self {
        CriteriaTypes(types.map(|t| match t {
            CriterionType::Cost => CriterionType::Profit,
            CriterionType::Profit => CriterionType::Cost,
        }))
    }
}

impl Deref for CriteriaTypes {
    type Target = DVector<CriterionType>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for CriteriaTypes {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Index<usize> for CriteriaTypes {
    type Output = CriterionType;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl IndexMut<usize> for CriteriaTypes {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl TryFrom<&[i8]> for CriteriaTypes {
    type Error = ValidationError;

    fn try_from(slice: &[i8]) -> Result<Self, Self::Error> {
        CriteriaTypes::from_slice(slice)
    }
}
