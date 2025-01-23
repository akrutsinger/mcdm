//! Validation methods for bounded arrays and vectors.
use crate::errors::ValidationError;

use nalgebra::{DMatrix, DVector};

pub trait MatrixValidate {
    /// Validates the bounds for each criteria follows the form $[min, max]$.
    ///
    /// This function checks that the bounds matrix has exactly two columns and that, for each row,
    /// the first column (minimum value) is less than the second column (maximum value).
    ///
    /// # Returns
    ///
    /// * `Result<(), ValidationError>` - Returns `Ok(())` if the bounds matrix is valid.
    ///
    /// # Errors
    ///
    /// * [`ValidationError::InvalidShape`] - If the bounds matrix does not have exactly two
    ///   columns.
    /// * [`ValidationError::InvalidValue`] - If any row in the bounds matrix has a minimum value
    ///   that is not less than the maximum value.
    fn is_bounds_valid(&self) -> Result<(), ValidationError>;

    /// Validates that the reference ideal matrix is within the specified bounds.
    ///
    /// This function checks that each element of the reference ideal matrix is within the
    /// corresponding bounds specified in the bounds matrix. The bounds matrix should have two
    /// columns, where the first column represents the minimum values and the second column
    /// represents the maximum values.
    ///
    /// # Arguments
    ///
    /// * `bounds` - A `DMatrix<f64>` representing the bounds for each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<(), ValidationError>` - Returns `Ok(())` if the reference ideal matrix is within
    ///   the bounds.
    ///
    /// # Errors
    ///
    /// * [`ValidationError::InvalidShape`] - If the bounds matrix does not have exactly two
    ///   columns.
    /// * [`ValidationError::DimensionMismatch`] - If any element of the reference ideal matrix is
    ///   outside the specified bounds.
    fn is_reference_ideal_bounds_valid(&self, bounds: &DMatrix<f64>)
        -> Result<(), ValidationError>;

    /// Ensures that the decision matrix is within the specified bounds.
    ///
    /// This function checks that each element of the decision matrix is within the corresponding
    /// bounds specified in the bounds matrix. The bounds matrix should have two columns, where the
    /// first column represents the minimum values and the second column represents the maximum
    /// values.
    ///
    /// # Arguments
    ///
    /// * `bounds` - A `DMatrix<f64>` representing the bounds for each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<(), ValidationError>` - Returns `Ok(())` if the decision matrix is within the
    ///   bounds.
    ///
    /// # Errors
    ///
    /// * [`ValidationError::InvalidShape`] - If the bounds matrix does not have exactly two
    ///   columns.
    fn is_within_bounds(&self, bounds: &DMatrix<f64>) -> Result<(), ValidationError>;
}

pub trait VectorValidate {
    /// Validates that the expected solution point values lie within the specified bounds for each
    /// criterion.
    ///
    /// This function checks that each element of the expected solution point vector is within the
    /// corresponding bounds specified in the bounds matrix. The bounds matrix should have two
    /// columns, where the first column represents the minimum values and the second column
    /// represents the maximum values.
    ///
    /// # Arguments
    ///
    /// * `bounds` - A `DMatrix<f64>` representing the bounds for each criterion. The shape should
    ///   be $(n, 2)$ where $n$ is the number of criteria. Each row defines the minimum and maximum
    ///   values for a criterion and must be in the form $[min, max]$.
    ///
    /// # Returns
    ///
    /// * `Result<(), ValidationError>` - Returns `Ok(())` if the expected solution point values are
    ///   within the bounds.
    ///
    /// # Errors
    ///
    /// * [`ValidationError::DimensionMismatch`] - If the length of the expected solution point
    ///   vector does not match the number of rows in the bounds matrix.
    /// * [`ValidationError::InvalidValue`] - If any element of the expected solution point vector
    ///   is outside the specified bounds.
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
