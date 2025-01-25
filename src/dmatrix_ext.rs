use nalgebra::{DMatrix, DVector};

pub trait DMatrixExt {
    /// Weights each criteria of the matrix by the corresponding element in the vector.
    ///
    /// # Arguments
    /// * `weights` - A vector whose elements will be used to weight the corresponding criteria of
    ///   the matrix.
    ///
    /// # Panics
    /// This method panics if the number of columns in the matrix does not match the length of the vector.
    fn scale_columns(&self, vector: &DVector<f64>) -> DMatrix<f64>;
}

impl DMatrixExt for DMatrix<f64> {
    fn scale_columns(&self, weights: &DVector<f64>) -> DMatrix<f64> {
        assert_eq!(
            self.ncols(),
            weights.len(),
            "Matrix and weights must have the same number of columns"
        );
        let mut result = self.clone();
        for (m, _) in self.column_iter().enumerate() {
            result.column_mut(m).scale_mut(weights[m]);
        }
        result
    }
}
