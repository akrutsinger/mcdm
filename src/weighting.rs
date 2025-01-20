//! Methods for weighting a decision matrix.

use crate::errors::WeightingError;
use nalgebra::{DMatrix, DVector};

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
/// Here's an example of how to use the [`Weight`] trait with an [`Equal`](crate::weighting::Weight::weight_equal)
/// weighting scheme:
///
/// ```rust
/// use mcdm::weighting::Weight;
/// use nalgebra::dmatrix;
///
/// let decision_matrix = dmatrix![3.0, 4.0, 2.0; 1.0, 5.0, 3.0];
/// let weights = decision_matrix.weight_equal().unwrap();
/// println!("{}", weights);
/// ```
pub trait Weight {
    /// Calculates the weights for the given decision matrix using angular distance.
    ///
    /// The `Angular` method calculates the weights for each criterion using the angular distance
    /// between the criterion's values and the mean of the criterion's values. The weights are then
    /// normalized to sum to 1.0.
    ///
    /// # Weight Calculation
    ///
    /// Normalize the decision matrix using the [`Sum`](crate::normalization::Normalize::normalize_sum)
    /// normalization method.
    ///
    /// Next, calculate the angle between the criterion's values and the mean of the criterion's
    /// values.
    ///
    /// $$ \theta_j = \arccos \left(\frac{\sum_{i=1}^m \left(\frac{x_{ij}}{m}\right)}{\sqrt{\sum_{i=1}^m \left(x_{ij}\right)^2} \sqrt{\sum_{i=1}^m \left(\frac{1}{m}\right)^2}} \right) $$
    ///
    /// Then, normalize the angle by dividing it by the sum of the angles.
    ///
    /// $$ w_j = \frac{\theta_j}{\sum_{j=1}^m \theta_j} $$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the
    /// criterion (column) with $m$ alternatives.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2d decision matrix where rows represent alternatives and columns represent
    ///   criteria.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, WeightingError>` - A vector of weights for each criterion, or error
    ///   if the weighting calculation fails.
    fn weight_angular(&self) -> Result<DVector<f64>, WeightingError>;

    /// Calculates the weights for the given decision matrix using CRiteria Importance Through
    /// Intercriteria Correlation (CRITIC) method.
    ///
    /// The CRITIC method is based on two key concepts. Contrast intensity and conflict or
    /// correlation. Contrast intensity measures the amount of variation (or dispersion) in each
    /// criterion. A higher contrast meas the criterion discriminates better between the
    /// alternatives. Conflict or correlation measures the interrelationship between criteria.
    /// Criteria that are highly correlated (i.e., redundant) should have less weight compared to
    /// those that are independent or less correlated.
    ///
    /// # Weight Calculation
    ///
    /// Normalize the decision matrix using the [`MinMax`](crate::normalization::Normalize::normalize_min_max)
    /// normalization method.
    ///
    /// Then calculate the standard devision. This reflects the degree of contrast (variation) in
    /// that criterion. Criteria with a higher standard deviation are considered more important.
    ///
    /// $$ \sigma_j = \sqrt{\frac{\sum_{i=1}^m (x_{ij} - x_j)^2}{m}} \quad \text{for} \quad j=1, \ldots, n $$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
    /// (column) with $m$ alternatives and $n$ criteria.
    ///
    /// Next, compute the pearson correlation between the criteria:
    ///
    /// $$ r_{jk} = \frac{\sum_{i=1}^m (x_{ij} - x_j)(x_{ik} - x_k)}{\sqrt{\sum_{i=1}^m (x_{ij} - x_j)^2}\sqrt{\sum_{i=1}^m (x_{jk} - x_k)^2}} $$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the criterion
    /// (column) with $m$ alternatives and $n$ criteria.
    ///
    /// Next, compute the information content using both contrast intesity (standard devision) and its
    /// relationship with the other criteria (correlation):
    ///
    /// $$ C_j = \sigma_j \sum_{k=1}^n \left(1-r_{jk}\right) $$
    ///
    /// Finally, determine the objective weights
    ///
    /// $$ w_j = \frac{C_j}{\sum_{k=1}^n C_k} $$
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2d decision matrix where rows represent alternatives and columns represent
    ///   criteria.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, WeightingError>` - A vector of weights for each criterion, or error
    ///   if the weighting calculation fails.
    fn weight_critic(&self) -> Result<DVector<f64>, WeightingError>;

    /// A weighting method that assigns equal weights to all criteria in the decision matrix.
    ///
    /// The `Equal` method distriutes equal importance to each criterion in a decision matrix,
    /// regardless of its values. This method is useful when all criteria are assumed to be equally
    /// significant in the decision-making process.
    ///
    /// # Weight Calculation
    ///
    /// Given a decision matrix with $n$ criteria (columns), the equal weight for each criterion is
    /// calculated as:
    ///
    /// $$ w_i = \frac{1}{n} $$
    ///
    /// where $w_i$ is the weight assigned to criterion $i$, and $n$ is the number of criteria in
    /// the decision matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2d decision matrix where rows represent alternatives and columns represent
    ///   criteria.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, WeightingError>` - A vector of weights for each criterion, or error
    ///   if the weighting calculation fails.
    fn weight_equal(&self) -> Result<DVector<f64>, WeightingError>;

    /// Calculates the entropy-based weights for the given decision matrix.
    ///
    /// # Weight Calculation
    ///
    /// Normalize the decision matrix using the [`Sum`](crate::normalization::Normalize::normalize_sum)
    /// normalization method.
    ///
    /// Each criterion (column) in the normalized matrix is evaluated for its entropy. If any value
    /// in a column is zero, the entropy for that column is set to zero. Otherwise, the entropy is
    /// calculated by:
    ///
    /// $$E_j = -\frac{\sum_{i=1}^m r_{ij}\ln(r_{ij})}{\ln(m)}$$
    ///
    /// where $E_j$ is the entropy of column $j$, $m$ is the number of alternatives, $n$ is the
    /// number of criteria, and $r_{ij}$ is the value of normalized decision matrix for alternative
    /// $i$ and criterion $j$
    ///
    /// After calculating entropy measure $E_j$ for each criterion, derive the weights for each
    /// criterion:
    ///
    /// $$ w_j = \frac{1 - E_j}{\sum_{j=1}^n (1 - E_j)} \quad \text{for} \quad j=1, \ldots, n $$
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2d decision matrix where rows represent alternatives and columns represent
    ///   criteria.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, WeightingError>` - A vector of weights for each criterion, or error
    ///   if the weighting calculation fails.
    fn weight_entropy(&self) -> Result<DVector<f64>, WeightingError>;

    /// The Gini coefficient weighting method measures inequality or dispersion of a distribution.
    ///
    /// # Weight Calculation
    ///
    /// The Gini coefficient is used to calculate the weights for each criterion.
    ///
    /// $$ G_j = \frac{\sum_{i=1}^m \sum_{k=1}^m |r_{ij} - r_{kj}|}{2m^2 \bar{r}_j} $$
    ///
    /// where $G_j$ is the Gini coefficient for criterion $j$, $m$ is the number of alternatives,
    /// $r_{ij}$ and $r_{kj}$ are the value of the normalized decision matrix for alternatives $i$
    /// and $k$ criterion $j$, and $\bar{r}_j$ is the mean of the normalized values for criterion
    /// $j$.
    ///
    /// The Gini coefficient $G_j$ ranges from 0 to 1, where $G_j \approx 0$ is low dispersion where
    /// each alternative is equally important, and $G_j \approx 1$ is high dispersion meaning each
    /// alternative is very different from the others.
    ///
    /// After calculating the Gini coefficient $G_j$ for each criterion, derive the weights for each
    /// criterion:
    ///
    /// $$ w_j = \frac{G_j}{\sum_{j=1}^n G_j} \quad \text{for} \quad j=1, \ldots, n $$
    ///
    /// where $n$ is the number of criteria.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2d decision matrix where rows represent alternatives and columns represent
    ///   criteria.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, WeightingError>` - A vector of weights for each criterion, or error
    ///   if the weighting calculation fails.
    fn weight_gini(&self) -> Result<DVector<f64>, WeightingError>;

    /// Calculate weights using the Method Based on the Removal Effects of Criiteria (MEREC) method.
    ///
    /// # Weight Calculation
    ///
    /// Normalize the decision matrix using the [`Linear`](crate::normalization::Normalize::normalize_linear)
    /// method. When passing the `CriteriaTypes` to the `Linear` normalization method, you must
    /// switch the criteria types so each cost becomes a profit and each profit becomes a cost.
    ///
    /// The MEREC weighting method is based on the following formula:
    ///
    /// $$ S_i = \ln \left( 1 + \left ( \frac{1}{n}\sum_{j=1}^n \ln(r_{ij}) \right) \right) $$
    ///
    /// where $S_i$ is the overall performance of alternative $i$, $m$ is the number of alternatives,
    /// and $r_{ij}$ is the value of normalized decision matrix for alternative $i$ andcriterion
    /// $j$.
    ///
    /// Next, calculate the performance of alternatives after removal of each criterion:
    ///
    /// $$ S_i^{\prime} = \ln \left( 1 + \left(\frac{1}{n} \sum_{k,k \neq j}^n \left | \ln(r_{ik}) \right | \right) \right) $$
    ///
    /// Next, calculate the sum of absolute deviations using the following:
    ///
    /// $$ E_j = \sum_{i=1}^m S_{ij}^{\prime} - S_i $$
    ///
    /// Finally, calculate the weights using the following:
    ///
    /// $$ w_j = \frac{E_j}{\sum_{j=1}^n E_j} $$
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2d decision matrix where rows represent alternatives and columns represent
    ///   criteria.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, WeightingError>` - A vector of weights for each criterion, or error
    ///   if the weighting calculation fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use mcdm::normalization::Normalize;
    /// use mcdm::weighting::Weight;
    /// use mcdm::CriteriaTypes;
    /// use nalgebra::dmatrix;
    ///
    /// let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1]).unwrap();
    /// let normalized_matrix = matrix.normalize_linear(&criteria_types).unwrap();
    /// let weights = normalized_matrix.weight_merec().unwrap();
    /// ```
    fn weight_merec(&self) -> Result<DVector<f64>, WeightingError>;

    /// Calculates the weights for the given decision matrix using standard deviation.
    ///
    /// # Weight Calculation
    ///
    /// First calculate the standard deviation measure for all of the critera using:
    ///
    /// $$ \sigma_j = \sqrt{\frac{\sum_{i=1}^m (x_{ij} - x_j)^2}{m}} \quad \text{for} \quad j=1, \ldots, n $$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row) and $j$th elements of the
    /// criterion (column) with $m$ alternatives and $n$ criteria.
    ///
    /// Next, derive the weights based on the standard deviation of the criteria:
    ///
    /// $$ w_j = \frac{\sigma_j}{\sum_{j=1}^n \sigma_j} \quad \text{for} \quad j=1, \ldots, n $$
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2d decision matrix where rows represent alternatives and columns represent
    ///   criteria.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, WeightingError>` - A vector of weights for each criterion, or error
    ///   if the weighting calculation fails.
    fn weight_standard_deviation(&self) -> Result<DVector<f64>, WeightingError>;

    /// Calculates the weights for the given decision matrix using variance.
    ///
    /// Variance measures speread of dispersion of data points within a crition across alternatives.
    /// With this method, the underlying principle is that criterion with a higher variance has
    /// greater discriminative power, meaning it plays a more significant fole in distinguishing
    /// between alternatives. Therefore, criteria with higher variance are given more weight in the
    /// decision-making process.
    ///
    /// # Weight Calculation
    ///
    /// Normalize the decision matrix using the [`MinMax`](crate::normalization::Normalize::normalize_min_max) normalization
    /// method.
    ///
    /// Then calculate the variance measure for all of the criteria using:
    ///
    /// $$ \sigma_j^2 = \frac{1}{m} \sum_{i=1}^m (r_{ij} - \bar{r}_j)^2 \quad \text{for} \quad j=1, \ldots, n $$
    ///
    /// where $\sigma_j^2$ is the variance of criteria $j$, $r_{ij}$ is the value of the normalized
    /// decision matrix for the $i$th alternative (row) and $j$th criterion (column) with $m$
    /// alternatives and $n$ criteria, and $\bar{x}_j$ is the mean of the $j$th criterion values
    /// across all alternatives.
    ///
    /// Next, derive the weights based on the variance of the criteria:
    ///
    /// $$ w_j = \frac{\sigma_j^2}{\sum_{j=1}^n \sigma_j^2} \quad \text{for} \quad j=1, \ldots, n $$
    ///
    /// where $n$ is the number of criteria.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A 2d decision matrix where rows represent alternatives and columns represent
    ///   criteria.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, WeightingError>` - A vector of weights for each criterion, or error
    ///   if the weighting calculation fails.
    fn weight_variance(&self) -> Result<DVector<f64>, WeightingError>;
}

impl Weight for DMatrix<f64> {
    fn weight_angular(&self) -> Result<DVector<f64>, WeightingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        let inv_num_criteria = 1.0 / num_criteria as f64;

        let mut weights = DVector::zeros(num_criteria);
        let add_col = DVector::from_element(num_alternatives, inv_num_criteria);

        let add_col_norm = add_col.norm();

        for (i, vec) in self.column_iter().enumerate() {
            let numerator = vec.sum() * inv_num_criteria;
            let norm_vec = vec.norm();
            weights[i] = (numerator / (norm_vec * add_col_norm)).acos();
        }

        let weights_sum = weights.sum();
        let weights = weights / weights_sum;

        Ok(weights)
    }

    fn weight_critic(&self) -> Result<DVector<f64>, WeightingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        let column_means = self.row_mean();
        let column_stds = self.row_variance().map(|v| v.sqrt());

        let pearson_correlation = {
            let mut normalized_matrix = self.clone();
            for j in 0..num_criteria {
                let mut col = normalized_matrix.column_mut(j);
                for x in col.iter_mut() {
                    *x = (*x - column_means[j]) / column_stds[j];
                }
            }
            normalized_matrix.transpose() * &normalized_matrix / num_alternatives as f64
        };

        let correlation_information =
            &column_stds.component_mul(&pearson_correlation.map(|x| 1.0 - x).row_sum());
        let correlation_information_sum = correlation_information.sum();
        let weights = DVector::from_column_slice(
            (correlation_information / correlation_information_sum).as_slice(),
        );

        Ok(weights)
    }

    fn weight_equal(&self) -> Result<DVector<f64>, WeightingError> {
        let num_criteria = self.ncols();

        if num_criteria == 0 {
            return Err(WeightingError::ZeroRange);
        }

        let weight = 1.0 / num_criteria as f64;
        let weights = DVector::repeat(num_criteria, weight);

        Ok(weights)
    }

    fn weight_entropy(&self) -> Result<DVector<f64>, WeightingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        let mut entropies = DVector::zeros(num_criteria);

        // Iterate over all criteria in the normalized matrix
        for (j, col) in self.column_iter().enumerate() {
            let col_entropy: f64 = col.iter().filter(|&x| *x != 0.0).map(|&x| x * x.ln()).sum();

            entropies[j] = col_entropy;
        }

        let scale_factor = 1.0 / -(num_alternatives as f64).ln();
        entropies *= scale_factor;
        entropies.iter_mut().for_each(|x| *x = 1.0 - *x);

        Ok(entropies.clone() / entropies.sum())
    }

    fn weight_gini(&self) -> Result<DVector<f64>, WeightingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        // Initialize weights as a zero vector
        let mut weights = DVector::zeros(num_criteria);

        for (j, column) in self.column_iter().enumerate() {
            let column_mean = column.sum() / num_alternatives as f64;

            let numerator: f64 = column
                .iter()
                .map(|&x| column.iter().map(|&y| (x - y).abs()).sum::<f64>())
                .sum();

            let denominator = 2.0 * (num_alternatives as f64).powi(2) * column_mean;
            weights[j] = numerator / denominator;
        }

        // Normalize weights: weights / sum(weights)
        let weights_sum = weights.sum();
        Ok(weights / weights_sum)
    }

    fn weight_merec(&self) -> Result<DVector<f64>, WeightingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        let s = DVector::from_iterator(
            num_alternatives,
            self.row_iter().map(|row| {
                let log_values = row.iter().map(|&x| x.ln().abs()).sum::<f64>();
                (1.0 + log_values / num_criteria as f64).ln()
            }),
        );

        let mut s_prim = DMatrix::zeros(num_alternatives, num_criteria);

        for j in 0..num_criteria {
            let ex_nmatrix = self.clone().remove_column(j); // Remove column `j`

            for (i, row) in ex_nmatrix.row_iter().enumerate() {
                let log_values = row.iter().map(|&x| x.ln().abs()).sum::<f64>();
                s_prim[(i, j)] = (1.0 + log_values / num_criteria as f64).ln();
            }
        }

        let e = DVector::from_iterator(
            num_criteria,
            (0..num_criteria).map(|j| {
                s_prim
                    .column(j)
                    .iter()
                    .zip(s.iter())
                    .map(|(&s_prim_val, &s_val)| (s_prim_val - s_val).abs())
                    .sum::<f64>()
            }),
        );

        let e_sum = e.sum();
        Ok(e / e_sum)
    }

    fn weight_standard_deviation(&self) -> Result<DVector<f64>, WeightingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        let std = self.row_variance().map(|v| v.sqrt()).transpose();

        // Sum of the standard deviations
        let std_sum = std.sum();

        // Compute the normalized weights
        let weights = std / std_sum;

        Ok(weights)
    }

    fn weight_variance(&self) -> Result<DVector<f64>, WeightingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(WeightingError::EmptyMatrix);
        }

        let var = self.row_variance().transpose();
        let var_sum = var.sum();
        let weights = var / var_sum;

        Ok(weights)
    }
}
