//! Validation methods for bounded arrays and vectors.
use crate::errors::ValidationError;

use nalgebra::{DMatrix, DVector};

pub trait MatrixValidate {
    /// Validates the bounds for each criteria follows the form $[min, max]$.
    fn is_bounds_valid(&self) -> Result<(), ValidationError>;

    /// Validates the reference ideal array aligns with the given bounds for each criterion.
    ///
    /// Validations:
    /// 1. Ensure `ref_ideal` has a shape of `[M, 2]`, where `M` is the number of criteria.
    /// 2. Ensure shape of `ref_ideal` and `bounds` are the same.
    /// 3. Verifies all `ref_ideal` values lie within the `bounds` array and that the `ref_ideal` values
    ///    are in scending order (i.e., `[min, max]`).
    fn is_reference_ideal_bounds_valid(&self, bounds: &DMatrix<f64>)
        -> Result<(), ValidationError>;

    /// Ensures a DMatrix is within the specified bounds.
    fn is_within_bounds(&self, bounds: &DMatrix<f64>) -> Result<(), ValidationError>;
}

pub trait VectorValidate {
    fn is_expected_solution_point_in_bounds(
        &self,
        bounds: &DMatrix<f64>,
    ) -> Result<(), ValidationError>;
}

impl MatrixValidate for DMatrix<f64> {
    fn is_bounds_valid(&self) -> Result<(), ValidationError> {
        if self.ncols() != 2 {
            return Err(ValidationError::InvalidShape);
        }

        for row in self.row_iter() {
            let min = row[0];
            let max = row[1];

            if min >= max {
                return Err(ValidationError::InvalidValue);
            }
        }

        Ok(())
    }

    fn is_reference_ideal_bounds_valid(
        &self,
        bounds: &DMatrix<f64>,
    ) -> Result<(), ValidationError> {
        if self.ncols() != 2 {
            return Err(ValidationError::InvalidShape);
        }

        if self.shape() != bounds.shape() {
            return Err(ValidationError::DimensionMismatch);
        }

        for (i, row) in self.row_iter().enumerate() {
            let min = bounds[(i, 0)];
            let max = bounds[(i, 1)];

            if (row[0] < min || row[1] > max) || row[0] > row[1] {
                return Err(ValidationError::InvalidValue);
            }
        }

        Ok(())
    }

    fn is_within_bounds(&self, bounds: &DMatrix<f64>) -> Result<(), ValidationError> {
        if self.ncols() != bounds.nrows() {
            return Err(ValidationError::InvalidShape);
        }

        bounds.is_bounds_valid()?;

        for (i, col) in self.column_iter().enumerate() {
            let min = bounds[(i, 0)];
            let max = bounds[(i, 1)];

            if col.min() < min || col.max() > max {
                return Err(ValidationError::InvalidValue);
            }
        }

        Ok(())
    }
}

impl VectorValidate for DVector<f64> {
    /// Validates the expected solution point value lie within the specified bounds for each
    /// criterion.
    ///
    /// # Arguments
    ///
    /// * `bounds` - 2D matrix of the bounds for each criterion. The shape should be $(n, 2)$ where
    ///   $n$ is the number of criteria. Each row defines the minimum and maximum values for a
    ///   criterion and must be in the form $[min, max]$.
    fn is_expected_solution_point_in_bounds(
        &self,
        bounds: &DMatrix<f64>,
    ) -> Result<(), ValidationError> {
        bounds.is_bounds_valid()?;

        if self.len() != bounds.nrows() {
            return Err(ValidationError::DimensionMismatch);
        }

        for (i, row) in bounds.row_iter().enumerate() {
            let min = row[0];
            let max = row[1];

            if self[i] < min || self[i] > max {
                return Err(ValidationError::InvalidValue);
            }
        }

        Ok(())
    }
}