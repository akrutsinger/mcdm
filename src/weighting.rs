use crate::errors::WeightingError;
use ndarray::{Array1, Array2, Axis};
use ndarray_stats::CorrelationExt;

/// A trait for calculating weights in Multiple-Criteria Decision Making (MCDM) problems.
///
/// The [`Weight`] trait defines the method used to calculate the weight vector for the criteria in a
/// decision matrix. Weights represent the relative importance of each criterion in the
/// decision-making process and are used in conjunction with other MCDM techniques like
/// normalization and ranking.
///
/// Implementations of this trait can use different algorithms to compute the weights, depending on
/// the nature of the decision problem and the method being used.
///
/// # Required Methods
///
/// The [`Weight`] trait requires the implementation of the `weight` method, which generates a weight
/// vector based on the decision matrix provided.
///
/// # Example
///
/// Here's an example of how to use the [`Weight`] trait with an [`Equal`] weighting scheme:
///
/// ```rust
/// use mcdm::weighting::{Equal, Weight};
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

/// Calculates the weights for the given decision matrix using angular distance.
///
/// The `Angular` struct implements the `Weight` trait by calculating the weights for each
/// criterion using the angular distance between the criterion's values and the mean of the
/// criterion's values. The weights are then normalized to sum to 1.0.
///
/// # ⚠️ **Caution** ⚠️
/// This method expects the decision matrix be normalized using the [`Sum`](crate::normalization::Sum)
/// normalization method.
///
/// # Weight Calculation
///
/// Calculate the angle between the criterion's values and the mean of the criterion's values.
///
/// $$ \theta_j = \arccos \left(\frac{\sum_{i=1}^m \left(\frac{x_{ij}}{m}\right)}{\sqrt{\sum_{i=1}^m \left(x_{ij}\right)^2} \sqrt{\sum_{i=1}^m \left(\frac{1}{m}\right)^2}} \right) $$
///
/// Then, normalize the angle by dividing it by the sum of the angles.
///
/// $$ w_j = \frac{\theta_j}{\sum_{j=1}^m \theta_j} $$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
/// (column) with $m$ total criteria.
///
/// # Arguments
///
/// * `matrix` - A 2D array where rows represent alternatives and columns represent criteria.
///
/// # Returns
///
/// This method returns a `Result` containing an array of angular-based weights for each criterion.
/// On an error, this method returns [`WeightingError`].
pub struct Angular;

impl Weight for Angular {
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError> {
        let (n, m) = matrix.dim();

        if n == 0 || m == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        let mut weights: Vec<f64> = vec![0.0; m];
        let add_col = Array2::ones((n, 1)) / m as f64;

        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            let dot_product = col.sum() / m as f64;
            let norm_vec = col.mapv(|x| x.powi(2)).sum().sqrt();
            let norm_add_col = add_col.mapv(|x: f64| x.powi(2)).sum().sqrt();
            weights[i] = dot_product / (norm_vec * norm_add_col);
            weights[i] = weights[i].acos();
        }

        let sum_weights = weights.iter().sum::<f64>();
        weights.iter_mut().for_each(|x| *x /= sum_weights);
        Ok(weights.into())
    }
}

/// Calculates the weights for the given decision matrix using CRiteria Importance Through
/// Intercriteria Correlation (CRITIC) method.
///
/// The CRITIC method is based on two key concepts. Contrast intensity and conflict or correlation.
/// Contrast intensity measures the amount of variation (or dispersion) in each criterion. A higher
/// contrast meas the criterion discriminates better between the alternatives. Conflict or
/// correlation measures the interrelationship between criteria. Criteria that are highly correlated
/// (i.e., redundant) should have less weight compared to those that are independent or less
/// correlated.
///
/// # ⚠️ **Caution** ⚠️
/// This method expects the decision matrix be normalized using the [`MinMax`](crate::normalization::MinMax)
/// normalization method where all criteria are treated as profit types.
///
/// # Weight Calculation
///
/// Calculate the standard devision. This reflects the degree of contrast (variation) in that
/// criterion. Criteria with a higher standard deviation are considered more important.
///
/// $$ \sigma_j = \sqrt{\frac{\sum_{i=1}^m (x_{ij} - x_j)^2}{m}} \quad \text{for} \quad j=1, \ldots, n $$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
/// (column) with $m$ total criteria and $n$ total alternatives.
///
/// Next, compute the pearson correlation between the criteria:
///
/// $$ r_{jk} = \frac{\sum_{i=1}^m (x_{ij} - x_j)(x_{ik} - x_k)}{\sqrt{\sum_{i=1}^m (x_{ij} - x_j)^2}\sqrt{\sum_{i=1}^m (x_{jk} - x_k)^2}} $$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
/// (column) with $m$ total criteria and $n$ total alternatives.
///
/// Next, compute the information content using both contrast intesity (standard devision) and its
/// relationship with the other criteria (correlation):
///
/// $$ C_j = \sigma_j \sum_{k=1}^m \left(1-r_{jk}\right) $$
///
/// Finally, determine the objective weights
///
/// $$ w_j = \frac{C_j}{\sum_{k=1}^m C_k} $$
///
/// # Arguments
///
/// * `matrix` - A 2D array where rows represent alternatives and columns represent criteria.
///
/// # Returns
///
/// This method returns a `Result` containing an array of angular-based weights for each criterion.
/// On an error, this method returns [`WeightingError`].
pub struct Critic;

impl Weight for Critic {
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError> {
        let (n, m) = matrix.dim();

        if n == 0 || m == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        let std = matrix.std_axis(Axis(0), 1.0);
        // NOTE: ndarray-stat's pearson correlation works on the rows of the matrix, but we want to
        // work on the columns (criteria), so we transpose the matrix.
        let corr = matrix.t().pearson_correlation()?;
        let correlation_information = &std * corr.map(|x| 1.0 - x).sum_axis(Axis(0));
        let sum_correlation_information = correlation_information.sum();
        let weights = &correlation_information / sum_correlation_information;
        Ok(weights)
    }
}

/// A weighting method that assigns equal weights to all criteria in the decision matrix.
///
/// The [`Equal`] struct implements the [`Weight`] trait by distributing equal importance to each
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
/// (i.e., zero columns), it returns a [`WeightingError::ZeroRange`] error.
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
/// # ⚠️ **Caution** ⚠️
/// This method expects the decision matrix be normalized using the [`Sum`](crate::normalization::Sum) normalization method
/// *before* calling this function. Failure to do so may result in incorrect weights.
///
/// # Weight Calculation
///
/// Each criterion (column) in the normalized matrix is evaluated for its entropy. If any value in a
/// column is zero, the entropy for that column is set to zero. Otherwise, the entropy is calculated
/// by:
///
/// $$E_j = -\frac{\sum_{i=1}^m p_{ij}\ln(p_{ij})}{\ln(m)}$$
///
/// where $E_j$ is the entropy of column $j$, $m$ is the number of alternatives, $n$ is the number
/// of criteria, and $p_{ij}$ is the value of normalized decision matrix for criterion $j$ and
/// alternative $i$
///
/// After calculating entropy measure $E_j$ for each criterion, derive the weights for each criterion:
///
/// $$ w_j = \frac{1 - E_j}{\sum_{i=1}^n (1 - E_i)} \quad \text{for} \quad j=1, \ldots, n $$
///
/// # Arguments
///
/// * `matrix` - A 2D array where rows represent alternatives and columns represent criteria.
///
/// # Returns
///
/// This method returns a `Result` containing an array of entropy-based weights for each
/// criterion. On an error, this method returns [`WeightingError`].
pub struct Entropy;

impl Weight for Entropy {
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError> {
        let (num_alternatives, num_criteria) = matrix.dim();

        let mut entropies = Array1::zeros(num_criteria);

        // Iterate over all criteria in the normalized matrix
        for (i, col) in matrix.axis_iter(Axis(1)).enumerate() {
            if col.iter().all(|&x| x != 0.0) {
                let col_entropy = col.iter().map(|&x| x * x.ln()).sum();

                entropies[i] = col_entropy;
            }
        }

        entropies /= -(num_alternatives as f64).ln();
        entropies = 1.0 - entropies;

        Ok(entropies.clone() / entropies.sum())
    }
}

/// The Gini coefficient weighting method measures inequality or dispersion of a distribution.
///
/// # Weight Calculation
///
/// The Gini coefficient is used to calculate the weights for each criterion.
///
/// $$ G_j = \frac{\sum_{i=1}^n \sum_{k=1}^n |x_{ij} - x_{kj}|}{2n^2 \bar{x}_j} $$
///
/// where $G_j$ is the Gini coefficient for criterion $j$, $n$ is the number of alternatives,
/// $x_{ij}$ is the value of the decision matrix for criterion $j$ and alternative $i$, and
/// $\bar{x}_j$ is the mean of the values for the $j$th criterion across all alternatives.
///
/// After calculating the Gini coefficient $G_j$ for each criterion, derive the weights for each
/// criterion:
///
/// $$ w_j = \frac{G_j}{\sum_{i=1}^m G_i} \quad \text{for} \quad j=1, \ldots, m $$
///
/// where $m$ is the number of criteria.
///
/// # Arguments
///
/// * `matrix` - A 2D array where rows represent alternatives and columns represent criteria.
///
/// # Returns
///
/// This method returns a `Result` containing an array of Gini-based weights for each criterion.
/// On an error, this method returns [`WeightingError`].
pub struct Gini;

impl Weight for Gini {
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError> {
        let (n, m) = matrix.dim();
        let mut weights = Array1::zeros(m);
        for j in 0..m {
            let mut values = Array1::zeros(n);
            for i in 0..n {
                let sum_abs_diff = matrix
                    .column(j)
                    .iter()
                    .fold(0.0, |acc, &x| acc + (x - matrix[[i, j]]).abs());
                values[i] = sum_abs_diff
                    / (2.0
                        * n as f64
                        * n as f64
                        * matrix.column(j).mean().ok_or(WeightingError::EmptyInput(
                            ndarray_stats::errors::EmptyInput,
                        ))?);
            }
            weights[j] = values.sum();
        }

        let weights_sum = weights.sum();

        Ok(weights / weights_sum)
    }
}

/// Calculate weights using the Method Based on the Removal Effects of Criiteria (MEREC) method.
///
/// # ⚠️ **Caution** ⚠️
/// This method expects the decision matrix is normalized using the [`Linear`](crate::normalization::Linear)
/// normalization method *before* calling this function. Likewise, when passing the `CriteriaType`
/// to the `Linear` normalization method, you must flip the criteria types. I.e., each of the costs
/// are switch to profits and all profits are switched to costs. Failure to do so may result in unexpected
/// weights.
///
/// # Weight Calculation
///
/// The MEREC weighting method is based on the following formula:
///
/// $$ S_i = \ln \left( 1 + \left ( \frac{1}{m}\sum_j \ln(n^x_{ij}) \right) \right) $$
///
/// where $S_i$ is the overall performance of alternative $i$, $m$ is the number of alternatives,
/// and $n^x_{ij}$ is the value of normalized decision matrix for criterion $j$.
///
/// Next, calculate the performance of alternatives after removal of each criterion:
///
/// $$ S_i^{\prime} = \ln \left( 1 + \left(\frac{1}{m} \sum_{k,k \neq j} \left | \ln(n^x_{ij}) \right | \right) \right) $$
///
/// Next, calculate the sum of absolute deviations using the following:
///
/// $$ E_j = \sum_i S_{ij}^{\prime} - S_i $$
///
/// Finally, calculate the weights using the following:
///
/// $$ w_j = \frac{E_j}{\sum_k E_k} $$
///
/// # Arguments
///
/// * `matrix` - A normalized decision matrix.
///
/// # Returns
///
/// * `Result<Array1<f64>, WeightingError>` - A vector of weights for each criterion, or a
///   [`WeightingError`] if the computation fails.
///
/// # Example
///
/// ```rust
/// use mcdm::normalization::{Linear, Normalize};
/// use mcdm::weighting::{Merec, Weight};
/// use mcdm::CriteriaType;
/// use ndarray::array;
///
/// let matrix = array![[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]];
/// let criteria_types = CriteriaType::from(vec![-1, 1]).unwrap();
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
/// # Weight Calculation
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
/// criterion. On an error, this method returns [`WeightingError`].
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

/// Calculates the weights for the given decision matrix using variance.
///
/// Variance measures speread of dispersion of data points within a crition across alternatives.
/// With this method, the underlying principle is that criterion with a higher variance has greater
/// discriminative power, meaning it plays a more significant fole in distinguishing between
/// alternatives. Therefore, criteria with higher variance are given more weight in the
/// decision-making process.
///
/// # ⚠️ **Caution** ⚠️
/// This method expects the decision matrix be normalized using the [`MinMax`](crate::normalization::MinMax)
/// normalization method with all criteria as profits*before* calling this function. Failure to do
/// so may result in incorrect weights.
///
/// # Weight Calculation
///
/// First calculate the variance measure for all of the criteria using:
///
/// $$ V_j = \frac{1}{n} \sum_{i=1}^n (x_{ij} - \bar{x}_j)^2 \quad \text{for} \quad j=1, \ldots, m $$
///
/// where $x_{ij}$ is the value of the $i$th alternative (row) and $j$th criterion (column) with $m$
/// total criteria and $n$ total alternatives, and $\bar{x}_j$ is the mean of the $j$th criterion
/// values across all alternatives.
///
/// Next, derive the weights based on the variance of the criteria:
///
/// $$ w_j = \frac{V_j}{\sum_{k=1}^m V_k} \quad \text{for} \quad j=1, \ldots, m $$
///
/// where $m$ is the number of criteria.
///
/// # Arguments
///
/// * [matrix](cci:4://file:///home/austyn/code/mcdm/src/lib.rs:45:0-56:0) - A 2D array where rows represent alternatives and columns represent criteria.
///
/// # Returns
///
/// This method returns a `Result` containing an array of variance-based weights for each
/// criterion. On an error, this method returns [`WeightingError`].
pub struct Variance;

impl Weight for Variance {
    fn weight(matrix: &Array2<f64>) -> Result<Array1<f64>, WeightingError> {
        let var = matrix.var_axis(Axis(0), 1.0);
        let weights = var.clone() / var.sum();

        Ok(weights)
    }
}
