//! Extensions for the `DMatrix` type from the `nalgebra` crate.

use crate::errors::ValidationError;
use nalgebra::{DMatrix, DVector};

pub trait DMatrixExt {
    /// Weights each criteria of the matrix by the corresponding element in the vector.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector whose elements will be used to weight the corresponding criteria of
    ///   the matrix.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, RankingError>` - A weighted matrix if successful.
    ///
    /// # Errors
    ///
    /// * [`ValidationError::DimensionMismatch`] - If the number of weights do not match the number
    ///   of criteria.
    fn apply_column_weights(&self, vector: &DVector<f64>) -> Result<DMatrix<f64>, ValidationError>;
}

impl DMatrixExt for DMatrix<f64> {
    fn apply_column_weights(
        &self,
        weights: &DVector<f64>,
    ) -> Result<DMatrix<f64>, ValidationError> {
        if weights.len() != self.ncols() {
            return Err(ValidationError::DimensionMismatch);
        }

        let mut result = self.clone();
        for (m, _) in self.column_iter().enumerate() {
            result.column_mut(m).scale_mut(weights[m]);
        }

        Ok(result)
    }
}
