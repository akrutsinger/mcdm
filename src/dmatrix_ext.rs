use crate::CriteriaType;
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

    /// Returns a matrix with the ideal reference values for each criteria.
    ///
    /// The resulting matrix is an $n$-by-2 matrix where $n$ is the number of criteria. If the
    /// criterion is a cost, the ideal will be the minimium value of the associated bounds. If the
    /// criterion is a profit, the ideal is the maximum value of the associated bounds.
    ///
    /// # Arguments
    /// * `criteria_types` - A vector of `CriteriaType` values indicating the type of the criteria.
    fn get_ideal_from_bounds(&self, criteria_types: &[CriteriaType]) -> DMatrix<f64>;
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

    fn get_ideal_from_bounds(&self, criteria_types: &[CriteriaType]) -> DMatrix<f64> {
        let mut ideal: DMatrix<f64> = DMatrix::zeros(self.nrows(), self.ncols());

        // Each row is expected to represent [min, max] in an already validated bounds
        for (i, row) in self.row_iter().enumerate() {
            match criteria_types[i] {
                CriteriaType::Cost => {
                    ideal[(i, 0)] = row.min();
                    ideal[(i, 1)] = row.min();
                }
                CriteriaType::Profit => {
                    ideal[(i, 0)] = row.max();
                    ideal[(i, 1)] = row.max();
                }
            }
        }
        ideal
    }
}
