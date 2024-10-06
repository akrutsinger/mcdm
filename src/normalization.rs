use crate::errors::NormalizationError;
use crate::CriteriaType;
use ndarray::{Array2, Axis};

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
/// use mcdm::CriteriaType;
/// use mcdm::normalization::{MinMax, Normalize};
/// use ndarray::{array, Array1};
///
/// let decision_matrix = array![[4.0, 7.0, 8.0], [2.0, 9.0, 6.0], [3.0, 6.0, 9.0]];
/// let criteria_types = CriteriaType::from(vec![-1, 1, 1]).unwrap();
/// let normalized_matrix = MinMax::normalize(&decision_matrix, &criteria_types).unwrap();
/// println!("{:?}", normalized_matrix);
/// ```
///
/// # Arguments
///
/// * `matrix` - A 2D decision matrix (`Array2<f64>`), where each row represents an alternative and
///   each column represents a criterion.
/// * `types` - An array slice of &[[`CriteriaType`]] indicating the [`CriteriaType::Cost`] or
///   [`CriteriaType::Profit`] of each criterion. The length of this array must match the number of
///   columns in `matrix`.
///
/// # Returns
///
/// This method returns a `Result<Array2<f64>, NormalizationError>`, where:
///
/// - `Array2<f64>` is the normalized matrix with the same dimensions as the input matrix.
/// - [`NormalizationError`] is returned if the normalization process fails (e.g., due to a mismatch
///   in the dimensions of `matrix` and `types` or invalid input values).
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
    /// * `Result<Array2<f64>, NormalizationError>` - A normalized decision matrix, or an error
    ///   if the normalization fails (e.g., due to mismatched dimensions or invalid types).
    fn normalize(
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError>;
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
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = Array2::<f64>::zeros(matrix.raw_dim());

        // Iterate over each column (criterion)
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let min_value = col
                .into_iter()
                .copied()
                .reduce(f64::min)
                .ok_or(NormalizationError::NoMinimum)?;
            let max_value = col
                .into_iter()
                .copied()
                .reduce(f64::max)
                .ok_or(NormalizationError::NoMaximum)?;

            let col_sum = match types[i] {
                CriteriaType::Cost => col.mapv(|x| x - min_value).sum(),
                CriteriaType::Profit => col.mapv(|x| max_value - x).sum(),
            };

            for (j, value) in col.iter().enumerate() {
                if (*value).abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[[j, i]] = match types[i] {
                    CriteriaType::Cost => 1.0 - ((*value - min_value) / col_sum),
                    CriteriaType::Profit => 1.0 - ((max_value - *value) / col_sum),
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
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = Array2::<f64>::zeros(matrix.raw_dim());

        // Iterate over each column (criterion)
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let min_value = col
                .into_iter()
                .copied()
                .reduce(f64::min)
                .ok_or(NormalizationError::NoMinimum)?;
            let max_value = col
                .into_iter()
                .copied()
                .reduce(f64::max)
                .ok_or(NormalizationError::NoMaximum)?;

            // Avoid division by zero
            if (max_value).abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (j, value) in col.iter().enumerate() {
                if (*value).abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[[j, i]] = match types[i] {
                    CriteriaType::Cost => min_value / *value,
                    CriteriaType::Profit => *value / max_value,
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
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = Array2::<f64>::zeros(matrix.raw_dim());

        // Iterate over each column (criterion)
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let col_prod = col.product();

            for (j, value) in col.iter().enumerate() {
                if (*value).abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                let ln_ratio = (*value).ln() / col_prod.ln();

                normalized_matrix[[j, i]] = match types[i] {
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
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = Array2::<f64>::zeros(matrix.raw_dim());

        // Iterate over each column (criterion)
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let max_value = col
                .into_iter()
                .copied()
                .reduce(f64::max)
                .ok_or(NormalizationError::NoMaximum)?;

            // Avoid division by zero
            if (max_value).abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (j, value) in col.iter().enumerate() {
                normalized_matrix[[j, i]] = match types[i] {
                    CriteriaType::Cost => 1.0 - (*value / max_value),
                    CriteriaType::Profit => *value / max_value,
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
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = Array2::<f64>::zeros(matrix.raw_dim());

        // Iterate over each column (criterion)
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let min_value = col
                .into_iter()
                .copied()
                .reduce(f64::min)
                .ok_or(NormalizationError::NoMinimum)?;
            let max_value = col
                .into_iter()
                .copied()
                .reduce(f64::max)
                .ok_or(NormalizationError::NoMaximum)?;

            // Avoid division by zero
            if (max_value - min_value).abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (j, value) in col.iter().enumerate() {
                normalized_matrix[[j, i]] = match types[i] {
                    CriteriaType::Cost => (max_value - *value) / (max_value - min_value),
                    CriteriaType::Profit => (*value - min_value) / (max_value - min_value),
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
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = Array2::<f64>::zeros(matrix.raw_dim());

        // Iterate over each column (criterion)
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let min_value = col
                .into_iter()
                .copied()
                .reduce(f64::min)
                .ok_or(NormalizationError::NoMinimum)?;
            let max_value = col
                .into_iter()
                .copied()
                .reduce(f64::max)
                .ok_or(NormalizationError::NoMaximum)?;

            // Avoid division by zero
            if (max_value - min_value).abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (j, value) in col.iter().enumerate() {
                if (*value).abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[[j, i]] = match types[i] {
                    CriteriaType::Cost => (min_value / *value).powi(3),
                    CriteriaType::Profit => (*value / max_value).powi(2),
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
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = Array2::<f64>::zeros(matrix.raw_dim());

        // Iterate over each column (criterion)
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let col_sum = match types[i] {
                CriteriaType::Cost => col.mapv(|x| 1.0 / x).sum(),
                CriteriaType::Profit => col.sum(),
            };

            for (j, value) in col.iter().enumerate() {
                if (*value).abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[[j, i]] = match types[i] {
                    CriteriaType::Cost => (1.0 / *value) / col_sum,
                    CriteriaType::Profit => *value / col_sum,
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
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = Array2::<f64>::zeros(matrix.raw_dim());

        // Iterate over each column (criterion)
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let sqrt_col_sum = col.mapv(|x| x * x).sum().sqrt();

            for (j, value) in col.iter().enumerate() {
                if (*value).abs() < f64::EPSILON {
                    return Err(NormalizationError::ZeroRange);
                }

                normalized_matrix[[j, i]] = match types[i] {
                    CriteriaType::Cost => 1.0 - (*value / sqrt_col_sum),
                    CriteriaType::Profit => *value / sqrt_col_sum,
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
        matrix: &Array2<f64>,
        types: &[CriteriaType],
    ) -> Result<Array2<f64>, NormalizationError> {
        // Check if the matrix is not empty
        if matrix.is_empty() {
            return Err(NormalizationError::EmptyMatrix);
        }

        // Ensure enough criteria types for all criteria
        if types.len() != matrix.ncols() {
            return Err(NormalizationError::NormalizationCriteraTypeMismatch);
        }

        // Initialize a matrix to store the normalized values
        let mut normalized_matrix = Array2::<f64>::zeros(matrix.raw_dim());

        // Iterate over each column (criterion)
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let min_value = col
                .into_iter()
                .copied()
                .reduce(f64::min)
                .ok_or(NormalizationError::NoMinimum)?;
            let max_value = col
                .into_iter()
                .copied()
                .reduce(f64::max)
                .ok_or(NormalizationError::NoMaximum)?;

            // Avoid division by zero
            if (max_value).abs() < f64::EPSILON || (min_value).abs() < f64::EPSILON {
                return Err(NormalizationError::ZeroRange);
            }

            for (j, value) in col.iter().enumerate() {
                normalized_matrix[[j, i]] = match types[i] {
                    CriteriaType::Cost => 1.0 - ((min_value - *value).abs() / min_value),
                    CriteriaType::Profit => 1.0 - ((max_value - *value).abs() / max_value),
                };
            }
        }

        Ok(normalized_matrix)
    }
}
