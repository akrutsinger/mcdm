use crate::errors::WeightingError;
use crate::normalization::Normalize;
use crate::normalization::Sum;
use ndarray::{Array1, Array2, Axis};

/// A trait for calculating weights in Multiple-Criteria Decision Making (MCDM) problems.
///
/// The [Weight] trait defines the method used to calculate the weight vector for the criteria in a
/// decision matrix. Weights represent the relative importance of each criterion in the
/// decision-making process and are used in conjunction with other MCDM techniques like
/// normalization and ranking.
///
/// Implementations of this trait can use different algorithms to compute the weights, depending on
/// the nature of the decision problem and the method being used.
///
/// # Required Methods
///
/// The [Weight] trait requires the implementation of the `weight` method, which generates a weight
/// vector based on the decision matrix provided.
///
/// # Example
///
/// Here's an example of how to use the [Weight] trait with an [Equal] weighting scheme:
///
/// ```rust
/// use mcdm::{weights::{Equal, Weight}};
/// use ndarray::array;
///
/// let decision_matrix = array![[3.0, 4.0, 2.0], [1.0, 5.0, 3.0]];
/// let weights = Equal::weight(&decision_matrix).unwrap();
/// println!("{:?}", weights);
/// ```
///
/// # Arguments
///
/// * `matrix` - A reference to a 2D decision matrix (`Array2<f64>`), where rows represent
///   alternatives and columns represent criteria.
///
/// # Returns
///
/// This method returns a `Result<Array1<f64>, WeightingError>`, where:
///
/// * `Array1<f64>` is a 1D array of weights corresponding to each criterion.
/// * `WeightingError` is returned if an error occurs while calculating the weights (e.g., an
///   invalid matrix shape or calculation failure).
pub trait Weight {
    /// Calculate a weight vector for the criteria in the decision matrix.
    ///
    /// This method computes the relative importance (weights) of each criterion based on the
    /// provided decision matrix. The specific weighting technique used will depend on the
    /// implementation of the trait.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2d decision matrix (`Array2<f64>`) where rows represent alternative and
    ///   columns represent criteria.
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, WeightingError>` - A vector of weights for each criterion, or error
    ///   if the weighting calculation fails.
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError>;
}

/// A weighting method that assigns equal weights to all criteria in the decision matrix.
///
/// The [Equal] struct implements the [Weight] trait by distributing equal importance to each
/// criterion in a decision matrix, regardless of its values. This method is useful when all
/// criteria are assumed to be equally significant in the decision-making process.
///
/// # Weight Calculation
///
/// Given a decision matrix with $n$ criteria (columns), the equal weight for each criterion is
/// calculated as:
///
/// $$ w_i = \frac{1}{n} $$
///
/// where $w_i$ is the weight assigned to criterion $i$, and $n$ is the total number of
/// criteria in the decision matrix.
///
/// # Returns
///
/// This method returns an array of equal weights for each criterion. If the matrix has no criteria
/// (i.e., zero columns), it returns a [WeightingError::ZeroRange] error.
pub struct Equal;

impl Weight for Equal {
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError> {
        let num_criteria = matrix.ncols();

        if num_criteria == 0 {
            return Err(WeightingError::ZeroRange);
        }

        let weight = 1.0 / num_criteria as f64;
        let weights = Array1::from_elem(num_criteria, weight);

        Ok(weights)
    }
}

/// Calculates the entropy-based weights for the given decision matrix.
///
/// Before the weight can be calculated, the decision matrix is normalized using the [Sum]
/// normalization method.
///
/// Each criterion (column) in the normalized matrix is evaluated for its entropy. If any value in a
/// column is zero, the entropy for that column is set to zero. Otherwise, the entropy is calculated
/// by:
///
/// $$E_j -\frac{\sum_{i=1}^m p_{ij}\ln(p_{ij})}{\ln(m)}$$
///
/// where $E_j$ is the entropy of column $j$, $m$ is the number of alternatives, $n$ is the number
/// of criteria, and $p_{ij}$ is the value of normalized decision matrix for criterion $j$ and
/// alternative $i$
///
/// # Arguments
///
/// * `matrix` - A 2D array where rows represent alternatives and columns represent criteria.
///
/// # Returns
///
/// This method returns a `Result` containing an array of entropy-based weights for each
/// criterion. On an error, this method returns [WeightingError].
pub struct Entropy;

impl Weight for Entropy {
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError> {
        let (num_alternatives, num_criteria) = matrix.dim();

        let types = Array1::ones(num_criteria);
        let normalized_matrix = Sum::normalize(matrix, &types)?;
        let mut entropies = Array1::zeros(num_criteria);

        // Iterate over all criteria in the normalized matrix
        for (i, col) in normalized_matrix.axis_iter(Axis(1)).enumerate() {
            if col.iter().all(|&x| x != 0.0) {
                let col_entropy = col.iter().map(|&x| x * x.ln()).sum();

                entropies[i] = col_entropy;
            }
        }

        entropies /= (num_alternatives as f64).ln();

        Ok(entropies)
    }
}

/// Calculate weights using the MEREC method.
///
/// The Method Based on the Removal Effects of Criiteria (MEREC) method requires that the decision
/// matrix is normalized using a specific linear normalization technique *before* calling this
/// function. Failure to do so may result in incorrect weights.
///
/// # ⚠️ **Caution** ⚠️
/// Ensure that the decision matrix is normalized using the [Linear](crate::normalization::Linear)
/// normalization method applying the MEREC weighting method. This method assumes the matrix has
/// already been normalized.
///
/// # Arguments
///
/// * `matrix` - A normalized decision matrix.
///
/// # Returns
///
/// * `Result<Array1<f64>, WeightingError>` - A vector of weights for each criterion, or a
///   [WeightingError] if the computation fails.
///
/// # Example
///
/// ```rust
/// use mcdm::normalization::{Linear, Normalize};
/// use mcdm::weights::{Merec, Weight};
/// use ndarray::array;
///
/// let matrix = array![[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]];
/// let criteria_types = array![-1, 1];  // -1 for cost, 1 for profit
/// let normalized_matrix = Linear::normalize(&matrix, &criteria_types).unwrap();
/// let weights = Merec::weight(&normalized_matrix).unwrap();
/// ```
pub struct Merec;

impl Weight for Merec {
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError> {
        let (n, m) = matrix.dim();

        if n == 0 || m == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        // Step 1: Calculate S
        let log_matrix = matrix.mapv(|x| (x + f64::EPSILON).ln().abs()); // Add a small value (EPSILON) to avoid log(0)
        let mean_reference_entropy =
            (log_matrix.sum_axis(Axis(1)) / m as f64).mapv(|x| (1.0 + x).ln());

        // Step 2: Calculate S_prim by excluding each column in turn
        let mut modified_entropy = Array2::<f64>::zeros((n, m));

        for j in 0..m {
            // Create ex_matrix by removing the j-th column from the original matrix
            let ex_matrix = {
                let mut ex_matrix = Array2::zeros((n, m - 1));
                for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
                    let mut ex_row = ex_matrix.row_mut(i);
                    ex_row.assign(
                        &row.iter()
                            .enumerate()
                            .filter(|&(idx, _)| idx != j)
                            .map(|(_, &v)| v)
                            .collect::<Array1<f64>>(),
                    );
                }
                ex_matrix
            };

            let ex_log_matrix = ex_matrix.mapv(|x| (x + f64::EPSILON).ln().abs());
            let sum_ex_matrix =
                (ex_log_matrix.sum_axis(Axis(1)) / m as f64).mapv(|x| (1.0 + x).ln());

            modified_entropy.column_mut(j).assign(&sum_ex_matrix);
        }

        // Step 3: Calculate E as the absolute difference between modified_entropy and mean_reference_entropy
        let mut criterion_information_loss = Array1::<f64>::zeros(m);
        for j in 0..m {
            let diff = modified_entropy.column(j).to_owned() - &mean_reference_entropy;
            criterion_information_loss[j] = diff.mapv(f64::abs).sum();
        }

        // Step 4: Normalize the E vector to get the final weights
        let sum_e = criterion_information_loss.sum();
        let weights = criterion_information_loss.mapv(|e| e / sum_e);

        Ok(weights)
    }
}

/// Calculates the weights for the given decision matrix using standard deviation.
///
/// First calculate the standard deviation measure for all of the critera using:
///
/// $$ \sigma_j = \sqrt{\frac{\sum_{i=1}^m (x_{ij} - x_j)^2}{m}} \quad \text{for} \quad j=1, \ldots, n $$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
/// (column) with $m$ total criteria and $n$ total alternatives.
///
/// Next, derive the weights based on the standard deviation of the criteria:
///
/// $$ w_j = \frac{\sigma_j}{\sum_{j=1}^n \sigma_j} \quad \text{for} \quad j=1, \ldots, n $$
///
/// # Arguments
///
/// * `matrix` - A 2D array where rows represent alternatives and columns represent criteria.
///
/// # Returns
///
/// This method returns a `Result` containing an array of entropy-based weights for each
/// criterion. On an error, this method returns [WeightingError].
pub struct StandardDeviation;

impl Weight for StandardDeviation {
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError> {
        // Compute the standard deviation across columns (criteria).
        // NOTE: The Axis(0) here confused me for a long time, so I'm adding this note in hopes that
        // future me doesn't have to re-research what's going on here. Unlike `ndarray`'s
        // `axis_iter()`, where the axis refers to the dimension being iterated over, (e.g.,
        // `axis_iter(Axis(0))` means you're iterating over the rows of the matrix),
        // `std_axis(Axis(0))` means you're building an output array with the same number of columns
        // as the input because you're computing the standard deviation across the first element of
        // every row, then the second elemtn of each row, and so on. So, the output array will have
        // the same number of columns as the input. I don't know if this will really help future me;
        // thanks a lot for not really understanding past me...
        let std = matrix.std_axis(Axis(0), 1.0);

        // Sum of the standard deviations
        let std_sum = std.sum();

        // Compute the normalized weights
        let weights = std / std_sum;

        Ok(weights)
    }
}
