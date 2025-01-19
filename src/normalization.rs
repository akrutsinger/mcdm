//! Normalization methods for normalizing a decision matrix.

use crate::errors::NormalizationError;
use crate::CriteriaType;
use crate::MatrixValidate;
use nalgebra::{DMatrix, DVector};

/// A trait for normalizing decision matrices in Multiple-Criteria Decision Making (MCDM) problems.
///
/// The [`Normalize`] trait defines a method to transform a decision matrix into a normalized form,
/// where the values of the criteria are scaled to make them comparable. This is an essential step
/// in MCDM techniques, as the criteria can have different units, ranges, and orientations (e.g.,
/// profit or cost).
///
/// # Criteria Types
///
/// The normalization process requires an array of criteria types to indicate whether each criterion
/// is a **profit** or a **cost** criterion:
///
/// - [`CriteriaType::Profit`]: Higher values are preferred.
/// - [`CriteriaType::Cost`]: Lower values are preferred.
///
/// # Example
///
/// Here's an example of normalizing a decision matrix:
///
/// ```rust
/// use approx::assert_relative_eq;
/// use mcdm::CriteriaType;
/// use mcdm::normalization::Normalize;
/// use nalgebra::dmatrix;
///
/// let decision_matrix = dmatrix![
///     2.9, 2.31, 0.56, 1.89;
///     1.2, 1.34, 0.21, 2.48;
///     0.3, 2.48, 1.75, 1.69
/// ];
/// let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
/// let normalized_matrix = decision_matrix.normalize_min_max(&criteria_types).unwrap();
/// let expected_matrix = dmatrix![
///     0.0, 0.85087719, 0.22727273, 0.74683544;
///     0.65384615, 0.0, 0.0, 0.0;
///     1.0, 1.0, 1.0, 1.0
/// ];
/// assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
/// ```
pub trait Normalize {
    /// Considers the maximum and minimum values of the given criteria for normalization.
    ///
    /// For profit:
    /// $$r_{ij} = 1- \frac{\max_j(x_{ij}) - x_{ij}}{\sum_{i=1}^m(\max_j(x_{ij}) - x_{ij})}$$
    ///
    /// For cost:
    /// $$r_{ij} = 1- \frac{x_{ij} - \min_j(x_{ij})}{\sum_{i=1}^m(x_{ij} - \min_j(x_{ij})}$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the
    /// criterion (column), $\max_j$ is the maximum criterion value, and $\min_j$ is the minimum
    /// criterion value in the decision matrix with $m$ alternatives and $n$ criteria.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails (e.g., due to mismatched dimensions or invalid types).
    fn normalize_enhanced_accuracy(
        &self,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError>;

    /// Profit criteria depend on the critions maximum value and cost criteria depend on criterion
    /// minimum value.
    ///
    /// For profit:
    /// $$r_{ij} = \frac{x_{ij}}{\max_j(x_{ij})}$$
    ///
    /// For cost:
    /// $$r_{ij} = \frac{\min_j(x_{ij})}{x_{ij}}$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column), $\max_j$ is the maximum criterion value, and $\min_j$ is the minimum criterion value
    /// in the decision matrix.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_linear(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError>;

    /// Considers the natural logarithm of the product of the criterion values.
    ///
    /// For profit:
    /// $$r_{ij} = \frac{\ln(x_{ij})}{\ln(\prod_{i=1}^m x_{ij})}$$
    ///
    /// For cost:
    /// $$r_{ij} = \frac{1 - \frac{\ln(x_{ij})}{\ln(\prod_{i=1}^m x_{ij})}}{m - 1}$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the
    /// criterion (column) with $m$ alternatives.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_logarithmic(
        &self,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError>;

    /// Normalization function specific to the [`MARCOS`](crate::ranking::Rank::rank_marcos) ranking
    /// method.
    fn normalize_marcos(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError>;

    /// Consider maximum rating of criterion for a given criteria.
    ///
    /// For profit:
    /// $$r_{ij} = \frac{x_{ij}}{\max_j(x_{ij})}$$
    ///
    /// For cost:
    /// $$r_{ij} = 1 - \frac{x_{ij}}{\max_j(x_{ij})}$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column), and $\max_j$ is the maximum criterion value in the decision matrix.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_max(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError>;

    /// Considers the criterion's minimum and maximum value for normalization.
    ///
    /// For profit:
    /// $$r_{ij} = \frac{x_{ij} - \min_j(x_{ij})}{\max_j(x_{ij}) - \min_j(x_{ij})}$$
    ///
    /// For cost:
    /// $$r_{ij} = \frac{\max_j(x_{ij}) - x_{ij}}{\max_j(x_{ij}) - \min_j(x_{ij})}$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column), $\max_j$ is the maximum criterion value, and $\min_j$ is the minimum value in the
    /// decision matrix.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_min_max(&self, types: &[CriteriaType])
        -> Result<DMatrix<f64>, NormalizationError>;

    /// Considers exponentiation of criterion's minimum and maximum value for normalization.
    ///
    /// For profit:
    /// $$r_{ij} = \left(\frac{x_{ij}}{\max_j(x_{ij})}\right)^2$$
    ///
    /// For cost:
    /// $$r_{ij} = \left(\frac{\min_j(x_{ij})}{x_{ij}}\right)^3$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column), $\max_j$ is the maximum criterion value, and $\min_j$ is the minimum value in the
    /// decision matrix.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_nonlinear(
        &self,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError>;

    /// Normalization function specific to the [`OCRA`](crate::ranking::Rank::rank_ocra) ranking
    /// method.
    ///
    /// For profit:
    /// $$r_{ij} = \frac{x_{ij} - \min_j(x_{ij})}{\min_j(x_{ij})}$$
    ///
    /// For cost:
    /// $$r_{ij} = \frac{\max_j(x_{ij}) - x_{ij}}{\min_j(x_{ij})}$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column), $\max_j$ is the maximum criterion value, and $\min_j$ is the minimum value in the
    /// decision matrix.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_ocra(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError>;

    /// Normalization function specific to the [`RIM`](crate::ranking::Rank::rank_rim) ranking
    /// method.
    ///
    /// Start with an $m \times n$ decision matrix $X$:
    ///
    /// $$ X = [x_{ij}] =
    /// \begin{bmatrix}
    ///     x_{11} & x_{12} & \ldots & x_{1n} \\\\
    ///     x_{21} & x_{22} & \ldots & x_{2n} \\\\
    ///     \vdots & \vdots & \ddots & \vdots \\\\
    ///     x_{m1} & x_{m2} & \ldots & x_{mn}
    /// \end{bmatrix}
    /// $$
    ///
    /// Next, normalize $x_{ij}$ based on the the following cases:
    ///
    /// $$ f(x_{ij}, \[A,B\], \[C,D\]) = \begin{cases}
    ///     1 & \text{if } x_{ij} \in \[C,D\] \\\\
    ///     1 - \frac{d_{\min}(x_{ij}, \[C,D\])}{|A-C|} & \text{if } x_{ij} \in \[A,C\] \land A \neq C \\\\
    ///     1 - \frac{d_{\min}(x_{ij}, \[C,D\])}{|D-B|} & \text{if } x_{ij} \in \[D,B\] \land D \neq B
    /// \end{cases} $$
    ///
    /// where $\[A,B\]$ is the criteria range, $\[C,D\]$ is the reference ideal, and $x_{ij} \in \[A,B\],\[C,D\] \subset \[A,B\]$.
    /// The function $d_{\min}(x_{ij}, \[C,D\])$ is defined as:
    ///
    /// $$ d_{\min}(x, \[C,D\]) = \min(|x_{ij} - C|, |x_{ij} - D|) $$
    ///
    /// The normalization lets us map the value $x$ to the range $\[0,1\]$ in the criteria domain
    /// with regard to the ideal reference value. The RIM normalization is calculated as:
    ///
    /// $$ Y = \[y_{ij}\] = \[f(x\_{ij}, t_j, s_j)\] $$
    fn normalize_rim(
        &self,
        criteria_range: &DMatrix<f64>,
        reference_ideal: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, NormalizationError>;

    /// Normalization function specific to the [`SPOTIS`](crate::ranking::Rank::rank_spotis) ranking
    /// method.
    ///
    /// Calculate the normalized distance matrix. For each alternative
    /// $A_i, i \in \\{1, 2, \ldots, n\\}$, calculate its normalized distance with respect to the
    /// ideal solution for each criteria $C_j, j \in \\{1, 2, \ldots, m\\}$:
    ///
    /// $$ r_{ij} = \frac{A_{ij} - S_j^*}{S_j^{\max} - S_j^{\min}} $$
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    /// * `bounds` - A 2D array that defines the decision problembounds of the criteria. Each row
    ///   represent the bounds for a given criterion. The first value is the lower bound and the
    ///   second value is the upper bound. Row one corresponds to criterion one, and so on.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_spotis(
        &self,
        types: &[CriteriaType],
        bounds: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, NormalizationError>;

    /// Considers the sum of the criteria values for normalization.
    ///
    /// For profit:
    /// $$r_{ij} = \frac{x_{ij}}{\sum_{i=1}^m x_{ij}}$$
    ///
    /// For cost:
    /// $$r_{ij} = \frac{\frac{1}{x_{ij}}}{\sum_{i=1}^m \frac{1}{x_{ij}}}$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the
    /// criterion (column) with $m$ alternatives.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_sum(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError>;

    /// Considers the root of the sum of the squares of the criteria values for normalization.
    ///
    /// For profit:
    /// $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^m x^2_{ij}}}$$
    ///
    /// For cost:
    /// $$r_{ij} = 1 - \frac{x_{ij}}{\sqrt{\sum_{i=1}^m x^2_{ij}}}$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
    /// (column) with $m$ alternatives.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_vector(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError>;

    /// Normalization method proposed by Zavadskas and Turskis in 2008.
    ///
    /// For profit:
    /// $$r_{ij} = 1 - \frac{\left|\max_j(x_{ij}) - x_{ij}\right|}{\max_j(x_{ij})}$$
    ///
    /// For cost:
    /// $$r_{ij} = 1 - \frac{\left|\min_j(x_{ij}) - x_{ij}\right|}{\min_j(x_{ij})}$$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column), $\max_j$ is the maximum criterion value, and $\min_j$ is the minimum value in the
    /// decision matrix.
    ///
    /// # Arguments
    ///
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails.
    fn normalize_zavadskas_turskis(
        &self,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError>;
}

impl Normalize for DMatrix<f64> {
    fn normalize_enhanced_accuracy(
        &self,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let min_value = col.min();
            let max_value = col.max();

            let col_sum = match types[j] {
                CriteriaType::Cost => col.map(|x| x - min_value).sum(),
                CriteriaType::Profit => col.map(|x| max_value - x).sum(),
            };

            for (i, value) in col.iter().enumerate() {
                if value.abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => 1.0 - ((value - min_value) / col_sum),
                    CriteriaType::Profit => 1.0 - ((max_value - value) / col_sum),
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_linear(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let min_value = col.min();
            let max_value = col.max();

            // Avoid division by zero
            if max_value.abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (i, value) in col.iter().enumerate() {
                if value.abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => min_value / value,
                    CriteriaType::Profit => value / max_value,
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_logarithmic(
        &self,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let col_prod = col.product();

            for (i, value) in col.iter().enumerate() {
                if value.abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                let ln_ratio = (value).ln() / col_prod.ln();

                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => (1.0 - ln_ratio) / (self.nrows() as f64 - 1.0),
                    CriteriaType::Profit => ln_ratio,
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_marcos(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError> {
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        for (j, col) in self.column_iter().enumerate() {
            let next_to_last_row_value = self[(self.nrows() - 3, j)];
            // Avoid division by zero
            if next_to_last_row_value.abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (i, value) in col.iter().enumerate() {
                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => next_to_last_row_value / value,
                    CriteriaType::Profit => value / next_to_last_row_value,
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_max(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let max_value = col.max();

            // Avoid division by zero
            if max_value.abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (i, value) in col.iter().enumerate() {
                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => 1.0 - (value / max_value),
                    CriteriaType::Profit => value / max_value,
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_min_max(
        &self,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let min_value = col.min();
            let max_value = col.max();

            // Avoid division by zero
            if (max_value - min_value).abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (i, value) in col.iter().enumerate() {
                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => (max_value - value) / (max_value - min_value),
                    CriteriaType::Profit => (value - min_value) / (max_value - min_value),
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_nonlinear(
        &self,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let min_value = col.min();
            let max_value = col.max();

            // Avoid division by zero
            if (max_value - min_value).abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (i, value) in col.iter().enumerate() {
                if value.abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => (min_value / value).powi(3),
                    CriteriaType::Profit => (value / max_value).powi(2),
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_ocra(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let min_value = col.min();
            let max_value = col.max();

            // Avoid division by zero
            if (max_value - min_value).abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (i, value) in col.iter().enumerate() {
                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => (max_value - value) / min_value,
                    CriteriaType::Profit => (value - min_value) / min_value,
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_rim(
        &self,
        criteria_range: &DMatrix<f64>,
        reference_ideal: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, NormalizationError> {
        reference_ideal.is_reference_ideal_bounds_valid(criteria_range)?;

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        fn _dmin(x: f64, c: f64, d: f64) -> f64 {
            f64::min(f64::abs(x - c), f64::abs(x - d))
        }

        for (i, row) in self.row_iter().enumerate() {
            for (j, x) in row.iter().enumerate() {
                let a = criteria_range[(j, 0)];
                let b = criteria_range[(j, 1)];
                let c = reference_ideal[(j, 0)];
                let d = reference_ideal[(j, 1)];
                normalized_matrix[(i, j)] = if c <= *x && *x <= d {
                    1.0
                } else if (a <= *x && *x < c) && (a != c) {
                    1.0 - (_dmin(*x, c, d) / f64::abs(a - c))
                } else if (d < *x && *x <= b) && (d != b) {
                    1.0 - (_dmin(*x, c, d) / f64::abs(d - b))
                } else {
                    0.0
                }
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_spotis(
        &self,
        types: &[CriteriaType],
        bounds: &DMatrix<f64>,
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        let mut expected_solution_point = DVector::zeros(self.ncols());

        for (i, row) in bounds.row_iter().enumerate() {
            expected_solution_point[i] = match types[i] {
                CriteriaType::Cost => row.min(),
                CriteriaType::Profit => row.max(),
            }
        }

        let range = bounds.column(0) - bounds.column(1);

        let mut normalized_matrix = DMatrix::zeros(self.nrows(), self.ncols());

        for (i, row) in self.row_iter().enumerate() {
            let normalized_row = (row - &expected_solution_point.transpose())
                .component_div(&range.transpose())
                .map(|val: f64| val.abs());
            normalized_matrix.set_row(i, &normalized_row);
        }

        Ok(normalized_matrix)
    }

    fn normalize_sum(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let col_sum = match types[j] {
                CriteriaType::Cost => col.map(|x| 1.0 / x).sum(),
                CriteriaType::Profit => col.sum(),
            };

            for (i, value) in col.iter().enumerate() {
                if value.abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => (1.0 / value) / col_sum,
                    CriteriaType::Profit => value / col_sum,
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_vector(&self, types: &[CriteriaType]) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let sqrt_col_sum = col.map(|x| x * x).sum().sqrt();

            for (i, value) in col.iter().enumerate() {
                if value.abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => 1.0 - (value / sqrt_col_sum),
                    CriteriaType::Profit => value / sqrt_col_sum,
                };
            }
        }

        Ok(normalized_matrix)
    }

    fn normalize_zavadskas_turskis(
        &self,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if self.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != self.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(self.nrows(), self.ncols());

        // Iterate over each column (criterion)
        for (j, col) in self.column_iter().enumerate() {
            let min_value = col.min();
            let max_value = col.max();

            // Avoid division by zero
            if max_value.abs() < f64::EPSILON || min_value.abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (i, value) in col.iter().enumerate() {
                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => 1.0 - ((min_value - value).abs() / min_value),
                    CriteriaType::Profit => 1.0 - ((max_value - value).abs() / max_value),
                };
            }
        }

        Ok(normalized_matrix)
    }
}
