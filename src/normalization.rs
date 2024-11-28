//! Normalization methods for normalizing a decision matrix.

use crate::errors::NormalizationError;
use crate::CriteriaType;
use nalgebra::DMatrix;

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
/// use mcdm::normalization::{MinMax, Normalize};
/// use nalgebra::dmatrix;
///
/// let decision_matrix = dmatrix![
///     2.9, 2.31, 0.56, 1.89;
///     1.2, 1.34, 0.21, 2.48;
///     0.3, 2.48, 1.75, 1.69
/// ];
/// let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
/// let normalized_matrix = MinMax::normalize(&decision_matrix, &criteria_types).unwrap();
/// let expected_matrix = dmatrix![
///     0.0, 0.85087719, 0.22727273, 0.74683544;
///     0.65384615, 0.0, 0.0, 0.0;
///     1.0, 1.0, 1.0, 1.0
/// ];
/// assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
/// ```
pub trait Normalize {
    /// Normalizes the decision matrix incorporating the specified criteria types.
    ///
    /// This method transforms the decision matrix by scaling its values according to the type
    /// of each criterion (profit or cost). The goal is to make the criteria comparable by
    /// removing the impact of different units, ranges, or orientations.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The decision matrix to normalize. This should be a 2D array where rows
    ///   represent alternatives, and columns represent criteria.
    /// * `types` - An array slice indicating if each criterion is a profit or cost.
    ///
    /// # Returns
    ///
    /// * `Result<DMatrix<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails (e.g., due to mismatched dimensions or invalid types).
    fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError>;
}

/// Considers the maximum and minimum values of the given criteria for normalization.
///
/// For profit:
/// $$r_{ij} = 1- \frac{\max_j(x_{ij}) - x_{ij}}{\sum_{i=1}^m(\max_j(x_{ij}) - x_{ij})}$$
///
/// For cost:
/// $$r_{ij} = 1- \frac{x_{ij} - \min_j(x_{ij})}{\sum_{i=1}^m(x_{ij} - \min_j(x_{ij})}$$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
/// (column), $\max_j$ is the maximum criterion value, and $\min_j$ is the minimum criterion value
/// in the decision matrix with $m$ total criteria.
pub struct EnhancedAccuracy;

impl Normalize for EnhancedAccuracy {
    fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());

        // Iterate over each column (criterion)
        for (j, col) in matrix.column_iter().enumerate() {
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
}

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
pub struct Linear;

impl Normalize for Linear {
    fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());

        // Iterate over each column (criterion)
        for (j, col) in matrix.column_iter().enumerate() {
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
}

/// Considers the natural logarithm of the product of the criterion values.
///
/// For profit:
/// $$r_{ij} = \frac{\ln(x_{ij})}{\ln(\prod_{i=1}^m x_{ij})}$$
///
/// For cost:
/// $$r_{ij} = \frac{1 - \frac{\ln(x_{ij})}{\ln(\prod_{i=1}^m x_{ij})}}{m - 1}$$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
/// (column) with $m$ total alternatives.
pub struct Logarithmic;

impl Normalize for Logarithmic {
    fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());

        // Iterate over each column (criterion)
        for (j, col) in matrix.column_iter().enumerate() {
            let col_prod = col.product();

            for (i, value) in col.iter().enumerate() {
                if value.abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                let ln_ratio = (value).ln() / col_prod.ln();

                normalized_matrix[(i, j)] = match types[j] {
                    CriteriaType::Cost => (1.0 - ln_ratio) / (matrix.nrows() as f64 - 1.0),
                    CriteriaType::Profit => ln_ratio,
                };
            }
        }

        Ok(normalized_matrix)
    }
}

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
pub struct Max;

impl Normalize for Max {
    fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());

        // Iterate over each column (criterion)
        for (j, col) in matrix.column_iter().enumerate() {
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
}

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
pub struct MinMax;

impl Normalize for MinMax {
    fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());

        // Iterate over each column (criterion)
        for (j, col) in matrix.column_iter().enumerate() {
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
}

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
pub struct NonLinear;

impl NonLinear {
    pub fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());

        // Iterate over each column (criterion)
        for (j, col) in matrix.column_iter().enumerate() {
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
}

/// Considers the sum of the criteria values for normalization.
///
/// For profit:
/// $$r_{ij} = \frac{x_{ij}}{\sum_{i=1}^m x_{ij}}$$
///
/// For cost:
/// $$r_{ij} = \frac{\frac{1}{x_{ij}}}{\sum_{i=1}^m \frac{1}{x_{ij}}}$$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
/// (column) with $m$ total criteria.
pub struct Sum;

impl Normalize for Sum {
    fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());

        // Iterate over each column (criterion)
        for (j, col) in matrix.column_iter().enumerate() {
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
}

/// Considers the root of the sum of the squares of the criteria values for normalization.
///
/// For profit:
/// $$r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^m x^2_{ij}}}$$
///
/// For cost:
/// $$r_{ij} = 1 - \frac{x_{ij}}{\sqrt{\sum_{i=1}^m x^2_{ij}}}$$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
/// (column) with $m$ total criteria.
pub struct Vector;

impl Normalize for Vector {
    fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());

        // Iterate over each column (criterion)
        for (j, col) in matrix.column_iter().enumerate() {
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
}

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
pub struct ZavadskasTurskis;

impl ZavadskasTurskis {
    pub fn normalize(
        matrix: &DMatrix<f64>,
        types: &[CriteriaType],
    ) -> Result<DMatrix<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = DMatrix::<f64>::zeros(matrix.nrows(), matrix.ncols());

        // Iterate over each column (criterion)
        for (j, col) in matrix.column_iter().enumerate() {
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
