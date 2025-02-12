//! Techniques for ranking alternatives.

use crate::{CriteriaTypes, CriterionType, DMatrixExt, MatrixValidate, Normalize, RankingError};
use nalgebra::{DMatrix, DVector};

/// A trait for ranking alternatives in Multiple-Criteria Decision Making (MCDM).
///
/// The [`Rank`] trait defines a method used to rank alternatives based on a normalized or
/// un-normalized decision matrix and a set of weights for the criteria. The ranking process
/// evaluates how well each alternative performs across the criteria, considering the relative
/// importance of each criterion as given by the `weights` array.
///
/// Higher preference values indicate better alternatives. The specific ranking method used (such as
/// [`TOPSIS`](Rank::rank_topsis) or others) will depend on the implementation of this trait.
///
/// # Example
///
/// Here’s an example of ranking alternatives using the [`Rank`] trait:
///
/// ```rust
/// use mcdm::{CriteriaTypes, Rank};
/// use nalgebra::{dmatrix, dvector};
///
/// let normalized_matrix = dmatrix![0.8, 0.6; 0.5, 0.9; 0.3, 0.7];
/// let weights = dvector![0.6, 0.4];
/// let ranking = normalized_matrix.rank_topsis(&weights).unwrap();
/// println!("Ranking: {}", ranking);
/// ```
pub trait Rank {
    /// Ranks decision matrix alternatives using the Additive Ratio ASsessment (ARAS) method.
    ///
    /// The ARAS method expects the decision matrix before any normalization or manipulation. The
    /// method assesses alternatives by comparing their overall performance to the ideal (best)
    /// alternative. It calculates a utility degree for each alternative based on the ratio of the
    /// sum of weighted normalized values for each criterion relative to the ideal alternative,
    /// which has the maximum performance for each criterion.
    ///
    /// This method takes an $m{\times}n$ decision matrix
    ///
    /// $$ x_{ij} =
    /// \begin{bmatrix}
    /// x_{11} & x_{12} & \ldots & x_{1n} \\\\
    /// x_{21} & x_{22} & \ldots & x_{2n} \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// x_{m1} & x_{m2} & \ldots & x_{mn}
    /// \end{bmatrix}
    /// $$
    ///
    /// then extends the matrix by adding an additional "best case" alternative row based on the
    /// minimum or maximum values of each criterion column. If that criterion is a profit, we use
    /// the maximum; if the criterion is a cost, we use the minimum.
    ///
    /// $$ E =
    /// \begin{bmatrix}
    ///     E_0(x_{i1}) & E_0(x_{i2}) & \ldots & E_0(x_{in}) \\\\
    ///     x_{11} & x_{12} & \ldots & x_{1n} \\\\
    ///     x_{21} & x_{22} & \ldots & x_{2n} \\\\
    ///     \vdots & \vdots & \ddots & \vdots \\\\
    ///     x_{(m+1)1} & x_{(m+1)2} & \ldots & x_{(m+1)n}
    /// \end{bmatrix}
    /// $$
    ///
    /// where
    ///
    /// $$
    /// E_0(x_{i1}) = \begin{cases}
    ///     \max(x_{i1}) & \text{if } \text{criteria type} = \text{profit} \\\\
    ///     \min(x_{i1}) & \text{if } \text{criteria type} = \text{cost}
    /// \end{cases}
    /// $$
    ///
    /// Next, obtain the normalized matrix, $s_{ij}$ by using the [`Sum`](Normalize::normalize_sum)
    /// normalization method on $E$. Then compute the weighted matrix $v_{ij}$ using
    ///
    /// $$ v_{ij} = w_j s_{ij} $$
    ///
    /// Next, determine the optimal criterion values only for the extended "best case" alternative
    /// (remember, this is the first row of the extended matrix).
    ///
    /// $$ S_0 = \sum_{j=1}^n v_{0j} $$
    ///
    /// Likewise, determine the sum of each other alternative using
    ///
    /// $$ S_i = \sum_{j=1}^n v_{ij} $$
    ///
    /// Lastly, calculate the utility degree $K_i$ which determines the ranking of each alternative
    ///
    /// $$ K_i = \frac{S_i}{S_0} $$
    ///
    /// Alternatives with a higher $K_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_aras(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.49447117, 0.35767527, 1.0], epsilon = 1e-5);
    /// ```
    fn rank_aras(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the COmbined Compromise SOlution (COCOSO) method.
    ///
    /// The COCOSO method expects the decision matrix is normalized using the [`MinMax`](Normalize::normalize_min_max)
    /// method. Then calculates the weighted sum of the comparision sequence and the total power
    /// weight of the comparison sequence for each alternative. The values of $S_i$ are based on the
    /// grey relationship generation method and the values for $P_i$ are based on the multiplicative
    /// WASPAS method.
    ///
    /// $$ S_i = \sum_{j=1}^n(w_j r_{ij}) $$
    /// $$ P_i = \sum_{j=1}^n(r_{ij})^{w_j} $$
    ///
    /// where $S_i$ is the grey relationship, $P_i$ is the multiplicative `WASPAS`, $n$ is the
    /// number of criteria, $r_{ij}$ is the $i$th alternative and $j$th criterion of the normalized
    /// decision matrix, and $w_j$ is the $j$th weight.
    ///
    /// We then compute the relative weights of alternatives using aggregation strategies.
    ///
    /// $$ k_{ia} = \frac{P_i + S_i}{\sum_{i=1}^m \left(P_i + S_i\right)} $$
    /// $$ k_{ib} = \frac{S_i}{\min_i(S_i)} + \frac{P_i}{\min_i(P_i)} $$
    /// $$ k_{ic} = \frac{\lambda(S_i) + (1 - \lambda)(P_i)}{\lambda \max_i(S_i) + (1 - \lambda) \max_i(P_i)} \quad 0 \leq \lambda \leq 1 $$
    ///
    /// where $k_{ia}$ represents the average of the sums of [`WeightedSum`](Rank::rank_weighted_sum)
    /// and [`WeightedProduct`](Rank::rank_weighted_product) scores, $k_{ib}$ represents the
    /// [`WeightedSum`](Rank::rank_weighted_sum) and [`WeightedProduct`](Rank::rank_weighted_product)
    /// scores over the best scores for each each method respectfully, and $k_{ic}$ represents the
    /// [`WeightedSum`](Rank::rank_weighted_sum) and [`WeightedProduct`](Rank::rank_weighted_product)
    /// scores using the compromise strategy, and $m$ is the number of alternatives.
    ///
    /// Lastly, we rank the alternatives as follows:
    ///
    /// $$ k_i = (k_{ia}k_{ib}k_{ic})^{\frac{1}{3}} + \frac{1}{3}(k_{ia} + k_{ib} + k_{ic}) $$
    ///
    /// Alternatives with a higher $k_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    /// * `lambda` - An `Option<f64>` between 0.0 and 1.0 used to preference between scoring
    ///   strategies based on the grey relationship or the multiplicative `WASPAS` method. A value
    ///   less than 0.5 preferences the multiplicative `WASPAS` method. A value greater than 0.5
    ///   preferences the grey relationship. If `None` is provided, default value of 0.5 will be
    ///   used which prefers both strategies equally.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of weights does not match the number
    ///   of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Normalize, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_min_max(&criteria_types).unwrap();
    /// let lambda = 0.5; // Prefer both values equally
    /// let ranking = normalized_matrix.rank_cocoso(&weights, Some(lambda)).unwrap();
    /// assert_relative_eq!(ranking, dvector![3.24754746, 1.14396494, 5.83576765], epsilon = 1e-5);
    /// ```
    fn rank_cocoso(
        &self,
        weights: &DVector<f64>,
        lambda: Option<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the COmbinative Distance-based ASessment (CODAS) method.
    ///
    /// The CODAS method expects the decision matrix is normalized using the [`Linear`](Normalize::normalize_linear)
    /// method. Then calculates an assessment matrix based on the euclidean distance and taxicab
    /// distance from the negative ideal solution.
    ///
    /// Build a weighted matrix $v_{ij}$ using the normalized decision matrix, $r_{ij}$, and
    /// weights, $w_j$.
    ///
    /// $$ v_{ij} = r_{ij}{w_j} $$
    ///
    /// Next, determine the negative ideal solution (NIS) using the weighted matrix $v_{ij}$.
    ///
    /// $$ NIS_j = \min_{i=1}^n v_{ij} $$
    ///
    /// Calculate the euclidean distance and taxicab distance from the negative ideal solution
    ///
    /// $$ E_i = \sqrt{\sum_{i=1}^m(v_{ij} - NIS_j)^2} $$
    /// $$ T_i = \sum_{i=1}^m \left|v_{ij} - NIS_j\right| $$
    ///
    /// Next, build the assessment matrix
    ///
    /// $$ h_{ik} = (E_i - E_k) + (\psi(E_i - E_k) \times (T_i - T_k)) $$
    ///
    /// where $k \in \{1, 2, \ldots, n\}$ and $\psi$ is the threshold function to recognize the equality
    /// of the Euclidean distance of the two alternatives, defined as follows:
    ///
    /// $$ \psi(x) = \begin{cases} 1 & \text{if} & |x| \geq \tau \\\\ 0 & \text{if} & |x| \lt \tau \end{cases} $$
    ///
    /// where $\tau$ is the threshold value determined by the decisionmaker. Suggested values for $\tau$
    /// are between 0.01 and 0.05.
    ///
    /// Lastly, calculate the assessment score of each alternative
    ///
    /// $$ H_i = \sum_{k=1}^n h_{ik} \quad \text{for } i = 1, 2, \ldots, m $$
    ///
    /// Alternatives with a higher $H_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    /// * `tau` - The threshold value for the threshold function. Default is 0.02.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of weights does not match the number
    ///   of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Normalize, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_linear(&criteria_types).unwrap();
    /// let ranking = normalized_matrix.rank_codas(&weights, 0.02).unwrap();
    /// assert_relative_eq!(ranking, dvector![-0.40977725, -1.15891275, 1.56869], epsilon = 1e-5);
    /// ```
    fn rank_codas(&self, weights: &DVector<f64>, tau: f64) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the COmplex PRoportional ASsessment (COPRAS) method.
    ///
    /// The COPRAS method expects the decision matrix without normalization. This method evaluates
    /// alternatives by separately considering the effects of maximizing (beneficial) and minimizing
    /// (non-beneficial) index values of attributes. This approach allows COPRAS to assess the
    /// impact of each type of crition independently, ensuring both positive contributions and cost
    /// factors are accounted for in the final ranking. This separation provides a more balanced and
    /// accurate assessment of each alternative.
    ///
    /// Start by calculating the normalized decision matrix, $r_{ij}$, using the
    /// [`Sum`](Normalize::normalize_sum) method, but treat each criterion as a profit. The
    /// normalization is caclculated as:
    ///
    /// $$ r_{ij} = \frac{x_{ij}}{\sum_{i=1}^m x_{ij}} $$
    ///
    /// Next, build a weighted matrix $v_{ij}$ using the normalized decision matrix, $r_{ij}$, and
    /// weights.
    ///
    /// $$ v_{ij} = r_{ij}{w_j} $$
    ///
    /// Next, determine the sums of difficult normalized values of the weighted matrix $v_{ij}$.
    ///
    /// $$ S_{+i} = \sum_{j=1}^k v_{ij} $$
    /// $$ S_{-i} = \sum_{j=k+1}^n v_{ij} $$
    ///
    /// where $k$ is the number of attributes to maximize. The rest of the attributes from $k+1$ to
    /// $n$ are minimized. $S_{+i}$ and $S_{-i}$ show the level of the goal achievement for
    /// alternatives. Higher value of $S_{+i}$ indicates the alternative is better and a lower value
    /// of $S_{-i}$ indicate a better alternative.
    ///
    /// Next, calculate the relative significance of alternatives using:
    ///
    /// $$ Q_i = S_{+i} + \frac{S_{-\min} \sum_{i=1}^m S_{-i}}{S_{-i} \sum_{i=1}^m \left(\frac{S_{-\min}}{S_{-i}}\right)} $$
    ///
    /// Lastly, rank the alternatives using:
    ///
    /// $$ U_i = \frac{Q_i}{Q_i^{\max}} \times 100\\% $$
    ///
    /// Alternatives with a higher $U_i$ value are more preferred.
    ///
    /// where $Q_i^{\max}$ is the maximum value of the utility function. Better alternatives have
    /// higher $U_i$ values.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_copras(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![1.0, 0.6266752, 0.92104753], epsilon = 1e-5);
    /// ```
    fn rank_copras(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the Evaluation based on Distance from Average Solution (EDAS)
    /// method.
    ///
    /// The EDAS method ranks the alternatives using the average distance from the average solution.
    /// The method expects a decision matrix before normalization. We define the decision matrix as:
    ///
    /// $$ X_{ij} =
    /// \begin{bmatrix}
    ///     x_{11} & x_{12} & \ldots & x_{1n} \\\\
    ///     x_{21} & x_{22} & \ldots & x_{2n} \\\\
    ///     \vdots & \vdots & \ddots & \vdots \\\\
    ///     x_{m1} & x_{m2} & \ldots & x_{mn}
    /// \end{bmatrix}
    /// $$
    ///
    /// Then calculate the average solution for each criteria:
    ///
    /// $$ \overline{X}\_{ij} = \frac{\sum_{i=1}^{m} x_{ij}}{m} $$
    ///
    /// Next, calculate the positive and negative distance from the mean solution for each
    /// alternative. When the criteria type is profit, compute the positive and negative distance
    /// as:
    ///
    /// $$ PD_{i} = \frac{\max(0, (X_{ij} - \overline{X}\_{ij}))}{\overline{X}\_{ij}} $$
    /// $$ ND_{i} = \frac{\max(0, (\overline{X}\_{ij} - X_{ij}))}{\overline{X}\_{ij}} $$
    ///
    /// When the criteria type is cost, compute the positive and negative distance as:
    ///
    /// $$ PD_{i} = \frac{\max(0, (\overline{X}\_{ij} - X_{ij}))}{\overline{X}\_{ij}} $$
    /// $$ ND_{i} = \frac{\max(0, (X_{ij} - \overline{X}\_{ij}))}{\overline{X}\_{ij}} $$
    ///
    /// Next, calculate the weighted sums for $PD$ and $ND$:
    ///
    /// $$ SP_i = \sum_{j=1}^{n} w_j PD_{ij} $$
    /// $$ SN_i = \sum_{j=1}^{n} w_j ND_{ij} $$
    ///
    /// Next, normalize the weighted sums:
    ///
    /// $$ NSP_i = \frac{SP_i}{\max_i(SP_i)} $$
    /// $$ NSN_i = 1 - \frac{SN_i}{\max_i(SN_i)} $$
    ///
    /// Finally, rank the alternatives by calculating their evaluation scores as:
    ///
    /// $$ E_i = \frac{NSP_i + NSN_i}{2} $$
    ///
    /// Alternatives with a higher $E_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_edas(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.04747397, 0.04029913, 1.0], epsilon = 1e-5);
    /// ```
    fn rank_edas(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the Enhanced Relative Value Decomposition (ERVD) method.
    ///
    /// The ERVD method expects the decision matrix is not normalized.
    ///
    /// To evaluate decision alternatives of an $m \times n$ unnormalized decision matrix with $m$
    /// alternatives and $n$ decision criteria.
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
    /// Then, define the reference points $\mu,j=1,\ldots,n$ for each decision criterion.
    ///
    /// Next, normalize the decision matrix using the [`Sum`](Normalize::normalize_sum) method
    /// treating all criteria as profits to get the normalized decision matrix $r_{ij}$.
    ///
    /// $$r_{ij} = \frac{x_{ij}}{\sum_{i=1}^m x_{ij}}$$
    ///
    /// Next, transform the reference points into the normalized scale:
    ///
    /// $$ \varphi_j = \frac{\mu_j}{\sum_{i=1}^m x_{ij}} $$
    ///
    /// where $\mu_j$ is the $j$th reference point vector and $x_{ij}$ is the $i$th alternative
    /// (row) and $j$th criteria (column) of the decision matrix.
    ///
    /// Next, calculate the reference value decision according to criterion $C_j$ by the
    /// increasing value function for profit criteria:
    ///
    /// $$ v_{ij} = \begin{cases}
    ///     (r_{ij} - \varphi_j)^\alpha & \text{if} \quad r_{ij} > \varphi_j \\\\
    ///     \text{-}\lambda(\varphi_j - r_{ij})^\alpha & \text{otherwise}
    /// \end{cases} $$
    ///
    /// and decreasing value function for cost criteria:
    ///
    ///$$ v_{ij} = \begin{cases}
    ///     (\varphi_j - r_{ij})^\alpha & \text{if} \quad r_{ij} < \varphi_j \\\\
    ///     \text{-}\lambda(r_{ij} - \varphi_j)^\alpha & \text{otherwise}
    /// \end{cases} $$
    ///
    ///
    /// Next, determine the positive ideal solution (PIS), $A^+$, and negative ideal solutions
    /// (NIS), $A^-$ using:
    ///
    /// $$ A^+ = \left\\{ v_1^+, \ldots, v_n^+ \right\\} $$
    /// $$ A^- = \left\\{ v_1^-, \ldots, v_n^- \right\\} $$
    ///
    /// where $v_j^+ = \max_i(v_{ij})$ and $v_j^- = \min_i(v_{ij})$.
    ///
    /// Next, calculate the separation measures, $S_i$, from PIS and NIS individually using the
    /// [Minkowski metric](https://en.wikipedia.org/wiki/Minkowski_distance):
    ///
    /// $$ S_i^+ = \sum_{j=1}^n w_j \cdot \left| v_{ij} - v_j^+ \right| $$
    /// $$ S_i^- = \sum_{j=1}^n w_j \cdot \left| v_{ij} - v_j^- \right| $$
    ///
    /// Finally, rank the alternatives by calculating their relative closeness to the ideal
    /// solution:
    ///
    /// $$ \phi_i = \frac{S_i^-}{S_i^+ + S_i^-} \quad \text{for} \quad i=1, \ldots, m$$
    ///
    /// Alternatives with a higher $\phi_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    /// * `reference_point` - A vector of reference points representing to the relative importance
    ///   of each criterion.
    /// * `lambda` - An `Option<f64>` scalar value representing attenuation factor for the losses.
    ///   Suggested to be between 2.0 and 2.5. If `None` is passed, the default value is 2.25.
    /// * `alpha` - An `Option<f64>` scalar value representing the diminishing sensitivity
    ///   parameter. If `None` is passed, the value defaults to 0.88.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let reference_point = dvector![1.46666667, 2.04333333, 0.84, 2.02];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_ervd(
    ///     &criteria_types,
    ///     &weights,
    ///     &reference_point,
    ///     Some(0.5),
    ///     Some(0.5)
    /// ).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.30321682, 0.216203469, 1.0], epsilon = 1e-5);
    /// ```
    fn rank_ervd(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
        reference_point: &DVector<f64>,
        lambda: Option<f64>,
        alpha: Option<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the Multi-Attributive Border Approximation Area Comparison
    /// (MABAC)
    /// method.
    ///
    /// The MABAC method expects an $m \times n$ decision matrix that is normalized using the
    /// [`MinMax`](Normalize::normalize_min_max) method. Then computes a
    /// weighted matrix $v_{ij}$ using
    ///
    /// $$ v_{ij} = {w_j}(r_{ij} + 1) $$
    ///
    /// where $r_{ij}$ is the $i$th alternative (row), $j$th criterion (column), and $w_j$ is the
    /// weight of the $j$th criterion.
    ///
    /// Next, calculate the boundary appromixation area for all criteria:
    ///
    /// $$ g_j = \left( \prod_{i=1}^m v_{ij} \right)^{1/m} $$
    ///
    /// where $g_j$ is the boundary approximation area for the $i$th alternative, $v_{ij}$ is the
    /// weighted matrix for the $i$th alternative and $j$th criterion, and $m$ is the number of
    /// alternatives.
    ///
    /// Next, calculate the distance of the $i$th alternative and $j$th criterion from the
    /// boundary approximation area
    ///
    /// $$ q_{ij} = v_{ij} - g_j \quad \text{for } i=1, \ldots, m \text{, } j=1, \ldots, n$$
    ///
    /// Lastly, rank the alternatives according to the sum of the distances of the alternatives
    /// from the border approximation area.
    ///
    /// $$ S_i = \sum_{j=1}^{m} q_{ij} \quad \text{for } i=1, \ldots, m \text{, } j=1, \ldots, n $$
    ///
    /// where $q_{ij}$ is the distance of the $i$th alternative and $j$th criterion of the weighted
    /// matrix $v_{ij}$ to the boundary approximation $g_j$, $m$ is the number of alternatives and
    /// $n$ is the number of criteria.
    ///
    /// Alternatives with a higher $S_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Normalize, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_min_max(&criteria_types).unwrap();
    /// let ranking = normalized_matrix.rank_mabac(&weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![-0.01955314, -0.31233795,  0.52420052], epsilon = 1e-5);
    /// ```
    fn rank_mabac(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError>;

    /// Ranks alternatives using the Multi-Attributive Ideal-Real Comparative Analysis (MAIRCA)
    /// method.
    ///
    /// The MAIRCA method operates on a normalized decision matrix. The typical normalization method
    /// used is the [`MinMax`](Normalize::normalize_min_max) method.
    ///
    /// Start with a normalied $m \times n$ decision matrix where $m$ is the number of alternatives
    /// and $n$ is the number of criteria.
    ///
    /// $$ r_{ij} =
    /// \begin{bmatrix}
    /// x_{11} & x_{12} & \ldots & x_{1n} \\\\
    /// x_{21} & x_{22} & \ldots & x_{2n} \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// x_{m1} & x_{m2} & \ldots & x_{mn}
    /// \end{bmatrix}
    /// $$
    ///
    /// Next, calculate the preference for choosing alternatives using the vector $P_{Ai}$ where
    ///
    /// $$ P_{Ai} = \frac{1}{m} $$
    ///
    /// Next, calculate a theoretical ranking matrix $T_p$ where
    ///
    /// $$ T_p =
    /// \begin{bmatrix}
    /// t_{p11} & t_{p12} & \ldots & t_{p1n} \\\\
    /// t_{p21} & t_{p22} & \ldots & t_{p2n} \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// t_{pm1} & t_{pm2} & \ldots & t_{pmn}
    /// \end{bmatrix} =
    /// \begin{bmatrix}
    /// P_{A1} \cdot w_1 & P_{A1} \cdot w_2 & \ldots & P_{A1} \cdot w_n \\\\
    /// P_{A2} \cdot w_1 & P_{A2} \cdot w_2 & \ldots & P_{A2} \cdot w_n \\\\
    /// \vdots           & \vdots           & \ddots & \vdots \\\\
    /// P_{Am} \cdot w_1 & P_{Am} \cdot w_2 & \ldots & P_{Am} \cdot w_n
    /// \end{bmatrix}
    /// $$
    ///
    /// Next, calculate the real rating matrix
    ///
    /// $$ T_r =
    /// \begin{bmatrix}
    /// t_{r11} & t_{r12} & \ldots & t_{r1n} \\\\
    /// t_{r21} & t_{r22} & \ldots & t_{r2n} \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// t_{rm1} & t_{rm2} & \ldots & t_{rmn}
    /// \end{bmatrix}
    /// $$
    ///
    /// The values of the real rating matrix are dependent on the criteria type. If the criteria
    /// type is profit:
    ///
    /// $$ t_{rij} = t_{pij} \cdot \left(  \frac{r_{ij} - \min(r_j)}{\max(r_j) - \min(r_j)} \right) $$
    ///
    /// if the criteria type is cost:
    ///
    /// $$ t_{rij} = t_{pij} \cdot \left(  \frac{r_{ij} - \max(r_j)}{\min(r_j) - \max(r_j)} \right) $$
    ///
    /// Next, calculate the total gap matrix, $G$, by taking the element-wise difference between the
    /// theoretical ranking method and the real rating matrix.
    ///
    /// $$ G = T_p - T_r  =
    /// \begin{bmatrix}
    /// t_{p11} - t_{r11} & t_{p12} - t_{r12} & \ldots & t_{p1n} - t_{r1n} \\\\
    /// t_{p21} - t_{r21} & t_{p22} - t_{r22} & \ldots & t_{p2n} - t_{r2n} \\\\
    /// \vdots            & \vdots            & \ddots & \vdots \\\\
    /// t_{pm1} - t_{rm1} & t_{pm2} - t_{rm2} & \ldots & t_{pmn} - t_{rmn}
    /// \end{bmatrix}
    /// $$
    ///
    /// Finally, rank the alternatives using the sum of the rows of the gap matrix, $G$.
    ///
    /// $$ Q_i = \sum_{j=1}^n g_{ij} $$
    ///
    /// Alternatives with a higher $Q_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of weights does not match the number
    ///   of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Normalize, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_min_max(&criteria_types).unwrap();
    /// let ranking = normalized_matrix.rank_mairca(&weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.18125122, 0.27884615, 0.0], epsilon = 1e-5);
    /// ```
    fn rank_mairca(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError>;

    /// Ranks alternatives using the Measurement of Alternatives and Ranking according to COmpromise
    /// Solutions (MARCOS) method.
    ///
    /// The MARCOS method is designed to rank alternatives of a non-normalized decision matrix based
    /// on their performance across multiple criteria. It combines normalization, ideal solutions,
    /// and compromise approaches to evaulate alternatives.
    ///
    /// We start by defining an augmented decision matrix, $M$, with $m$ rows of alternatives and
    /// $n$ columns of criteria. The first row of $M$ defines the anti-ideal ($aam$) solution, and
    /// the last row defines the ideal ($aim$) solution.
    ///
    /// $$ M = \begin{bmatrix}
    /// x_{aa1} & x_{aa2} & \ldots & x_{aan} \\\\
    /// x_{11} & x_{12} & \ldots & x_{1n} \\\\
    /// x_{21} & x_{22} & \ldots & x_{2n} \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// x_{m1} & x_{m2} & \ldots & x_{mn} \\\\
    /// x_{ai1} & x_{ai2} & \ldots & x_{aim} \\\\
    /// \end{bmatrix}
    /// $$
    ///
    /// The ideal solution ($AI$) and anti-ideal solution ($AAI$) values for cost ($C$) and profit
    /// ($P$) criteria are defined as:
    ///
    /// $$ AI = \begin{cases}
    ///     \max_i(x_{ij}) & \text{if } j \in P \\\\
    ///     \min_i(x_{ij}) & \text{if } j \in C
    /// \end{cases} $$
    ///
    /// $$ AAI = \begin{cases}
    ///     \min_i(x_{ij}) & \text{if } j \in P \\\\
    ///     \max_i(x_{ij}) & \text{if } j \in C
    /// \end{cases} $$
    ///
    /// Next, normalize the decision matrix using the following:
    ///
    /// $$ r_{ij} = \begin{cases}
    ///     \frac{x_{ij}}{x_{aij}} & \text{if } j \in P \\\\
    ///     \frac{x_{aij}}{x_{ij}} & \text{if } j \in C
    /// \end{cases} $$
    ///
    /// where $x_{ij}$ is the value of alternative $i$ in criterion $j$ and $x_{ai}$ is the value of
    /// the ideal solution for criterion $j$.
    ///
    /// Next, calculate the weighted matrix as:
    ///
    /// $$ v_{ij} = w_j \cdot (r_{ij} + 1) $$
    ///
    /// Next, calculate the degrees of utility, $K_i$ of each alternative where $K_i^+$ is the ideal
    /// solution and $K_i^-$ is the anti-ideal solution:
    ///
    /// $$ K_i^+ = \frac{S_i}{S_{ai}} $$
    /// $$ K_i^- = \frac{S_i}{S_{aai}} $$
    ///
    /// where $S_i (i = 1, \ldots, m)$ represents the sum of the elemnts of the weighted matrix $V$
    /// as represented by:
    ///
    /// $$ S_i = \sum_{i=1}^m v_{ij} $$
    ///
    /// Next, calculate the utility of each alternative where $f(K_i^+)$ is the utility of the ideal
    /// solution and $f(K_i^-)$ is the utility of the anti-ideal solution defined as:
    ///
    /// $$ f(K_i^+) = \frac{K_i^+}{K_i^+ + K_i^-} $$
    /// $$ f(K_i^-) = \frac{K_i^-}{K_i^+ + K_i^-} $$
    ///
    /// Finally, calculate ranking by making a determination from the utility functions:
    ///
    /// $$ f(K_i) = \frac{K_i^+ + K_i^-}{1 + \frac{1 - f(K_i^+)}{f(K_i^+)} + \frac{1 - f(K_i^-)}{f(K_i^-)}} $$
    ///
    /// Alternatives with a lower $f(K_i)$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// Finally, calculate the preference values:
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_marcos(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.51306940, 0.36312213, 0.91249658], epsilon = 1e-5);
    /// ```
    fn rank_marcos(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks alternatives using the Multi-Objective Optimization on the basis of Ratio Analysis
    /// (MOORA) method.
    ///
    /// The MOORA ranking method expects an $m \times n$ unnormalized decision matrix. This method
    /// ranks alternatives by optimizing multiple objectives, such as maximizing benefits and
    /// minimizing costs.
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
    /// Next, treat all criteria types as profit and normalize the decision matrix using the
    /// [`Vector`](Normalize::normalize_vector) method:
    ///
    /// $$ r_{ij} = \frac{x_{ij}}{\sqrt{\sum_{i=1}^m x^2_{ij}}} $$
    ///
    /// where $x_{ij}$ is the $i$th alternative (row) and $j$th criterion (column).
    ///
    /// Next, calculate the weighted matrix as:
    ///
    /// $$ v_{ij} = w_j \cdot r_{ij} $$
    ///
    /// Next, calculate the sums of the profit, $S_p$, and cost criteira, $S_c$:
    ///
    /// $$ S_p = \sum_{j=1}^n v_{ij} \quad \text{if } j \text{ is a profit criterion} $$
    /// $$ S_c = \sum_{j=1}^n v_{ij} \quad \text{if } j \text{ is a cost criterion} $$
    ///
    /// Finally, calculate the composite score $y_i$ as:
    ///
    /// $$ y_i = S_p - S_c $$
    ///
    /// Alternatives with a higher $y_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_moora(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![-0.12902028, -0.14965973,  0.26377149], epsilon = 1e-5);
    /// ```
    fn rank_moora(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks alternatives using the Operational Competitiveness Rating Analysis (OCRA) method.
    ///
    /// The OCRA ranking method expects an $m \times n$ unnormalized decision matrix. This method
    /// ranks alternatives by comparing their performance across multiple criteria. It is designed
    /// to handle both beneficial critieria (those to be maximized, such as profits or quality)
    /// and non-beneficial criteria (those to be minimized, such as costs or environmental impact).
    ///
    /// Start by normalizing the decision matrix, $x_{ij}$, using the [`OCRA`](Normalize::normalize_ocra)
    /// method:
    ///
    /// For profit:
    /// $$r_{ij} = \frac{x_{ij} - \min_j(x_{ij})}{\min_j(x_{ij})}$$
    ///
    /// For cost:
    /// $$r_{ij} = \frac{\max_j(x_{ij}) - x_{ij}}{\min_j(x_{ij})}$$
    ///
    /// Next, determine the preferences for profit, $\bar{\bar{I_i}}$, and cost, $\bar{\bar{O_i}}$:
    ///
    /// $$ \bar{\bar{I_i}} = \bar{I_i} - \min(\bar{I_i}) $$
    /// $$ \bar{\bar{O_i}} = \bar{O_i} - \min(\bar{O_i}) $$
    ///
    /// where $\bar{I_i}$ is a measure of relative performance for the $i$th alternative and cost
    /// criteria, and $\bar{O_i}$ is a measure of relative cost for the $i$th alternative and profit
    /// criteria.
    ///
    /// Finally, determine the overall preference, $P_i$ of the alternatives:
    ///
    /// $$ P_i = \left( \bar{\bar{I_i}} + \bar{\bar{O_i}} \right) - \min\left( \bar{\bar{I_i}} + \bar{\bar{O_i}} \right) $$
    ///
    /// Alternatives with a higher $P_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_ocra(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.0, 0.73175174, 3.64463555], epsilon = 1e-5);
    /// ```
    fn rank_ocra(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Rank alternatives using Preference Ranking on the Basis of Ideal-Average Distance (PROBID)
    /// method.
    ///
    /// This ranking method expects an $m \times n$ unnormalized decision matrix. It ranks
    /// alternatives based on their proximity to an ideal solution while considering average
    /// performance of alternatives.
    ///
    /// Start with a decision matrix, $x_{ij}$, where $i$ represents alternatives and $j$ represents
    /// criteria.
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
    /// Normalize the decision matrix using the [`Vector`](Normalize::normalize_vector) method to
    /// get the normalized matrix, $r_{ij}$.
    ///
    /// Next, calculate the normalized weighted decision matrix, $v_{ij}$:
    ///
    /// $$ v_{ij} = w_j \cdot r_{ij} $$
    ///
    /// where $w_j$ is the weight of the $j$th criterion.
    ///
    /// Next, sort the normalized weighted decision matrix by criteria and criteria type. This will
    /// create a matrix of successive Positive Ideal Solutions. Represent the sorted normalized
    /// weighted decision matrix, $A_{(k)}$ as:
    ///
    /// $$ \begin{split}
    /// A_{(k)} &= \left\\{ (\text{Largest}(v_j, k) | j \in J), \ldots, (\text{Smallest}(v_j, k) | j \in J^\prime) \right\\} \\\\
    ///         &= \left\\{ v_{(k)1}, v_{(k)2}, \ldots, v_{(k)n} \right\\}
    /// \end{split} $$
    ///
    /// where $k \in \\{1, 2, \ldots, m\\}$, $J$ is the set of profit criteria, and $J^\prime$ is
    /// the set of cost criteria.
    ///
    /// Next, find the average value of each objective column as:
    ///
    /// $$ \bar{v}\_j = \frac{\sum_{k=1}^m v_{(k)j}}{m} \quad \text{for } j \in \\{1,2,\ldots,n\\} $$
    ///
    /// and the average solution given by:
    ///
    /// $$ \bar{A} = \left( \bar{v}\_1, \bar{v}\_2, \ldots, \bar{v}\_n \right) $$
    ///
    /// Next, iteratively calculate the distance of each solution to the ideal solutions and the
    /// average solution. The distance to the ideal solutions is:
    ///
    /// $$ S_{i(k)} = \sqrt{\sum_{j=1}^n \left( v_{ij} - v\_{(k)j} \right)^2} \quad \text{for } i \in \\{1,2,\ldots,m\\} $$
    ///
    /// Next, determind overall positive ideal distance, $S_i^+$, and negative ideal distance,
    /// $S_i^-$ as:
    ///
    /// $$ S_i^+ = \begin{cases}
    ///     \sum_{k=1}^{(m+1)/2} \frac{1}{k} S_{i(k)} & i \in \\{1,2,\ldots,m\\} \text{when } m \text{ is odd} \\\\
    ///     \sum_{k=1}^{m/2} \frac{1}{k} S_{i(k)} & i \in \\{1,2,\ldots,m\\} \text{when } m \text{ is even}
    /// \end{cases} $$
    ///
    /// $$ S_i^- = \begin{cases}
    ///     \sum_{k=(m+1)/2}^m \frac{1}{m-k+1} S_{i(k)} & i \in \\{1,2,\ldots,m\\} \text{when } m \text{ is odd} \\\\
    ///     \sum_{k=m/2+1}^m \frac{1}{m-k+1} S_{i(k)} & i \in \\{1,2,\ldots,m\\} \text{when } m \text{ is even}
    /// \end{cases} $$
    ///
    /// For the simpler PROBID method, the distance calculate is slightly different:
    ///
    /// $$ S_i^+ = \begin{cases}
    ///     \sum_{k=1}^{m/4} \frac{1}{k} S_{i(k)} & i \in \\{1,2,\ldots,m\\} \text{when } m \geq 4 \\\\
    ///     S_{i(1)} & i \in \\{1,2,\ldots,m\\} \text{when } 0<m<4
    /// \end{cases} $$
    ///
    /// $$ S_i^- = \begin{cases}
    ///     \sum_{k=m+1-(m/4)}^m \frac{1}{m-k+1} S_{i(k)} & i \in \\{1,2,\ldots,m\\} \text{when } m \text{ is odd} \\\\
    ///     S_{i(m)} & i \in \\{1,2,\ldots,m\\} \text{when } 0<m<4
    /// \end{cases} $$
    ///
    /// Next, calculate the positive ideal to negative ideal ration, $R_i$:
    ///
    /// $$ R_i = \frac{S_i^+}{S_i^-} $$
    ///
    /// Finally, calculate the performance score as:
    ///
    /// $$ P_i = \frac{1}{1 + R_i^2} + S_{i(\text{avg})} $$
    ///
    /// Alternatives with a higher $P_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    /// * `simpler_probid` - A boolean value indicating whether to use the simpler PROBID method.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_probid(&criteria_types, &weights, false).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.31041914, 0.39049427, 1.1111652], epsilon = 1e-5);
    /// let ranking = matrix.rank_probid(&criteria_types, &weights, true).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.0, 1.31254211, 3.3648893], epsilon = 1e-5);
    /// ```
    fn rank_probid(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
        simpler_probid: bool,
    ) -> Result<DVector<f64>, RankingError>;

    /// Rank alternatives using the Root Assessment Method (RAM).
    ///
    /// The RAM method expects an $m \times n$ decision matrix normalized using the
    /// [`Sum`](Normalize::normalize_sum) method. This method ranks alternatives by comparing them
    /// to a root or reference alternative. RAM considers both the relative performance of
    /// alternatives and their deviation from the reference point, ensuring a balanced evaluation
    /// across multiple criteria.
    ///
    /// Start with a normalized decision matrix $r_{ij}$. The [`Sum`](Normalize::normalize_sum)
    /// method is typically used to normalize the decision matrix.
    ///
    /// Next, calculate the weighted decision matrix:
    ///
    /// $$ v_{ij} = r_{ij} w_j $$
    ///
    /// Next, calculate the sums of weighted normalized scored for the profit ($y_{ij}^+$) and cost
    /// ($y_{ij}^-$) criteria of the $i$th alternative:
    ///
    /// $$ S_i^+ = \sum_{j=1}^n y_{ij}^+ $$
    ///
    /// $$ S_i^- = \sum_{j=1}^n y_{ij}^- $$
    ///
    /// Finally, compute the performance score as:
    ///
    /// $$ P_i = \sqrt[{2 + S_i^-}]{2 + S_i^+}$$
    ///
    /// Alternatives with higher $P_i$ values are more preferred.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_ram(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![1.40671879, 1.39992479, 1.48267751], epsilon = 1e-5);
    /// ```
    fn rank_ram(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Rank alternatives using the Reference Ideal Method (RIM).
    ///
    /// The RIM method expects a non-normalized decision matrix and uses criteria bounds and
    /// reference ideal to evaluate alternatives.
    ///
    /// Term definitions:
    ///
    /// * Criteria weights: $w_j$, $j \in \\{1, 2, \ldots, n\\}$ where $n$ is the number of
    ///   criteria.
    /// * Decision matrix: An unnormalized $m \times n$ matrix $X = [x_{ij}]$ where $m$ is the
    ///   number of alternatives, $n$ is the number of criteria, and $x_{ij}$ is the value of the
    ///   $j$th criterion for the $i$th alternative.
    /// * Criteria Range: $t_j = [t_j^{(\min)}, t_j^{(\max)}]$, $j \in \\{1, 2, \ldots, n\\}$. This
    ///   defines an artibrary bounds for each criterion.
    /// * Reference Ideal: $s_j = [s_j^{(\min)}, s_j^{(\max)}]$, $j \in \\{1, 2, \ldots, n\\}$ where
    ///   $[s_j^{(\min)}, s_j^{(\max)}] \subset [t_j^{(\min)}, t_j^{(\max)}]$. The Reference Ideal
    ///   defines the most preferred interval of values for each criterion. It can be derived from
    ///   the criteria range or defined as the expected outcome of the decision process.
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
    /// Next, normalize the decision matrix $X$ using the [`RIM`](Normalize::normalize_rim)
    /// normalization function:
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
    /// $$ d_{\min}(x_{ij}, \[C,D\]) = \min(|x_{ij} - C|, |x_{ij} - D|) $$
    ///
    /// The normalization lets us map the value $x_{ij}$ to the range $\[0,1\]$ in the criteria
    /// domain with regard to the ideal reference value. The RIM normalization is calculated as:
    ///
    /// $$ Y = \[y_{ij}\]= \[f(x\_{ij}, t_j, s_j)\] $$
    ///
    /// Next, calculate the weighted normalized matrix $Y\prime$:
    ///
    /// $$ Y^\prime = \[y^\prime_{ij}\]= \[w_j y\_{ij}\] $$
    ///
    /// Next, calculate the variation to the normalized reference ideal for each alternative $A_i$:
    ///
    /// $$ I^+_i = \sqrt{\sum\_{j=1}^n (y^\prime\_{ij} - w_j)^2}$$
    ///
    /// $$ I^-_i = \sqrt{\sum\_{j=1}^n (y^\prime)^2}$$
    ///
    /// Lastly, calculate the relative index of each alternative:
    ///
    /// $$ P_i = \frac{I^+_i}{I^+_i + I^-_i} $$
    ///
    /// Alternatives with a higher $P_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    /// * `criteria_range` - A 2D array that defines the arbitrarily chosen bounds of the criteria.
    ///   Each row represent the bounds for a given criterion. The first value is the lower bound
    ///   and the second value is the upper bound. Row one corresponds to criterion one, and so on.
    /// * `reference_ideal` - A 2D array that defines the most preferred interval of values for each
    ///   criterion. Each row represent the bounds for a given criterion. The first value is the lower
    ///   bound and the second value is the upper bound. Row one corresponds to criterion one, and so
    ///   on. The `reference_ideal` must be a subset of the `criteria_range`.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let criteria_range = dmatrix![0.1, 3.0; 1.0, 2.48; 0.0, 1.9; 0.69, 3.1];
    /// let reference_ideal = dmatrix![0.1, 0.1; 2.48, 2.48; 1.9, 1.9; 0.69, 0.69];
    /// let ranking =
    ///     matrix.rank_rim(&criteria_types, &weights, &criteria_range, &reference_ideal).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.44909838, 0.33256763, 0.80336877], epsilon = 1e-5);
    /// ```
    fn rank_rim(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
        criteria_range: &DMatrix<f64>,
        reference_ideal: &DMatrix<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Rank alternatives using the Stable Preference Ordering Towards Ideal Solution (SPOTIS)
    /// method.
    ///
    /// The SPOTIS method expects an un-normalized decision matrix. This method evaluates given
    /// decision alternatives using the distance from the best ideal solution.
    ///
    /// Start with an $m \times n$ decision matrix of elements $x_{ij}$ where $m$ is the number of
    /// alternatives and $n$ is the number of criteria.
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
    /// Next, normalize the decision matrix using the [`SPOTIS`](Normalize::normalize_spotis)
    /// normalization method. This method define the bounds of the problem. Minimum and Maximum
    /// bounds of classical MCDM problems must be defined to transform MCDM problems from
    /// ill-defined to well-defined.
    ///
    /// $$ \left[ S_j^{\min}, S_j^{\max}\right], j \in \\{1, 2, \ldots, n\\} $$
    ///
    /// $S_j^*$ is the ideal solution point. This is a vector which includes maximum or minimum
    /// values from the bounds or specific criterion depending on the crition type. Take the maximum
    /// value for profit criteria and miimum value for cost criteria.
    ///
    /// $$ S_j^* = \begin{cases}
    ///     S_j^{\min} & \text{if } j \text{th criteria is cost} \\\\
    ///     S_j^{\max} & \text{if } j \text{th criteria is profit}
    /// \end{cases} $$
    ///
    /// The normalized matrix, $r_{ij}$ is the normalized distance matrix. For each alternative
    /// $A_i, i \in \\{1, 2, \ldots, m\\}$, calculate its normalized distance with respect to the
    /// ideal solution for each criteria $C_j, j \in \\{1, 2, \ldots, n\\}$:
    ///
    /// $$ r_{ij} = \frac{A_{ij} - S_j^*}{S_j^{\max} - S_j^{\min}} $$
    ///
    /// Finally, calculate the performance score as:
    ///
    /// $$ P_i = \sum_{j=1}^n w_j r_{ij} $$
    ///
    /// Alternatives with a lower $P_i$ values are more preferred.
    ///
    /// # Arguments
    ///
    /// * `criteria_types` - A [`CriteriaTypes`] indicating whether each criterion is a profit or
    ///   cost.
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    /// * `bounds` - A 2D array that defines the decision problembounds of the criteria. Each row
    ///   represent the bounds for a given criterion. The first value is the lower bound and the
    ///   second value is the upper bound. Row one corresponds to criterion one, and so on.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of criteria types or number of weights
    ///   do not match the number of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let bounds = dmatrix![0.1, 3.0; 1.0, 2.48; 0.0, 1.9; 0.69, 3.1];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_spotis(&criteria_types, &weights, &bounds).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.57089264, 0.69544822, 0.14071266], epsilon = 1e-5);
    /// ```
    fn rank_spotis(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
        bounds: &DMatrix<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the TOPSIS method.
    ///
    /// The TOPSIS method expects an $m \times n$ decision matrix normalized using the
    /// [`MinMax`](Normalize::normalize_min_max) method. Then computes a weighted matrix $v_{ij}$
    /// using
    ///
    /// $$ v_{ij} = r_{ij}{w_j} $$
    ///
    /// where $r_{ij}$ is the $i$th alternative (row), $j$th criterion (column) of the normalized
    /// decision matrix, and $w_j$ is the weight of the $j$th criterion.
    ///
    /// Next, calculate a positive ideal solution (PIS), $A_j^+$, and a negative ideal solution
    /// (NIS), $A_j^-$.
    ///
    /// $$ A^+ = \left\\{v_1^+, v_2^+, \dots, v_n^+\right\\} = \max_j v_{ij} $$
    /// $$ A^- = \left\\{v_1^-, v_2^-, \dots, v_n^-\right\\} = \min_j v_{ij} $$
    ///
    /// where $v_j^+ = \max_i(v_{ij})$ and $v_j^- = \min_i(v_{ij})$.
    ///
    /// Next, calculate the separation measures of each alternative to the PIS, $S_i^+$, and NIS,
    /// $S_i^-$. The separation measures are computed through Euclidean distance and calculated as:
    ///
    /// $$ S_i^+ = \sqrt{ \sum_{j=1}^{n} (v_{ij} - v_j^+)^2 } $$
    /// $$ S_i^- = \sqrt{ \sum_{j=1}^{n} (v_{ij} - v_j^-)^2 } $$
    ///
    /// Finally, calculate the relative closeness of each alternative ot the ideal solution.
    ///
    /// $$ \phi_i = \frac{S_i^-}{S_i^+ + S_i^-} $$
    ///
    /// Alternatives with a higher $\phi_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of weights does not match the number
    ///   of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    /// * [`RankingError::InvalidValue`] - If any of the weights are zero.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Normalize, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_min_max(&criteria_types).unwrap();
    /// let ranking = normalized_matrix.rank_topsis(&weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.52910451, 0.72983217, 0.0], epsilon = 1e-5);
    /// ```
    fn rank_topsis(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError>;

    /// Rank alternatives using the Weighted Aggregated Sum Product ASessment (WASPAS) method.
    ///
    /// The WASPAS method expects an $m \times n$ decision matrix normalized using the
    /// [`Linear`](Normalize::normalize_linear) method. The WASPAS ranking method is a combination
    /// of the [`WeightedProduct`](Rank::rank_weighted_product) and
    /// [`WeightedSum`](Rank::rank_weighted_sum) ranking methods.
    ///
    /// Start by calculating the [`WeightedProduct`](Rank::rank_weighted_product), $WPM$, and
    /// [`WeightedSum`](Rank::rank_weighted_sum), $WSM$.
    ///
    /// $$ WPM = \prod_{j=1}^n(r_{ij})^{w_j} $$
    /// $$ WSM = \sum_{j=1}^n r_{ij}{w_j} $$
    ///
    /// where $w_j$ is the weight of the $j$th criterion and $r_{ij}$ is the normalized decision
    /// matrix at the $i$th alternative and $j$th criterion.
    ///
    /// Lastly, calculate the total relative importance for each alternative:
    ///
    /// $$ \begin{split}
    /// Q_i &= \lambda WSM + (1-\lambda)WPM \\\\
    ///     &= \lambda \sum_{j=1}^n r_{ij}{w_j} + (1-\lambda) \prod_{j=1}^n(r_{ij})^{w_j}
    /// \end{split} $$
    ///
    /// Alternatives with a higher $Q_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    /// * `lambda` - An `Option<f64>` representing the preferance value between 0.0 and 1.0. If
    ///   `None`, the value defaults to 0.5 and represents equal preferance between WPM and WSM
    ///   ranking. A value less than 0.5 shifts preference in favor of WPM. A value greater than 0.5
    ///   shifts preference in favor of WSM ranking.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of weights does not match the number
    ///   of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    /// * [`RankingError::InvalidValue`] - If the lambda value is not between 0.0 and 1.0.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Normalize, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_type = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let lambda = 0.5;
    /// let normalized_matrix = matrix.normalize_linear(&criteria_type).unwrap();
    /// let ranking = normalized_matrix.rank_waspas(&weights, Some(lambda)).unwrap();
    /// assert_relative_eq!(
    ///     ranking,
    ///     dvector![0.48487887, 0.36106779, 1.0],
    ///     epsilon = 1e-5
    /// );
    /// ```
    fn rank_waspas(
        &self,
        weights: &DVector<f64>,
        lambda: Option<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Computes the Weighted Product Model (WPM) preference values for alternatives.
    ///
    /// The `WeightedProduct` model expects an $m \times n$ decision matrix normalized using the
    /// [`Sum`](Normalize::normalize_sum) method.
    ///
    /// Next, calculate the ranking using:
    ///
    /// $$ P_i = \prod_{j=1}^n(r_{ij})^{w_j} $$
    ///
    /// where $r_{ij}$ is the $i$th alternative (row) and $j$th criterion (column) of the normalized
    /// decision matrix with $n$ criteria, and $w_j$ is the weight of the $j$th criterion.
    ///
    /// Alternatives with a higher $P_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of weights does not match the number
    ///   of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Normalize, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_type = CriteriaTypes::from_slice(&[-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_sum(&criteria_type).unwrap();
    /// let ranking = normalized_matrix.rank_weighted_product(&weights).unwrap();
    /// assert_relative_eq!(
    ///     ranking,
    ///     dvector![0.21711531, 0.17273414, 0.53281425],
    ///     epsilon = 1e-5
    /// );
    /// ```
    fn rank_weighted_product(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError>;

    /// Rank the alternatives using the Weighted Sum Model.
    ///
    /// The `WeightedSum` method ranks alternatives based on the weighted sum of their criteria
    /// values. Each alternative's score is calculated by multiplying its criteria values by the
    /// corresponding weights and summing the results. This method expects an $m \times n$ decision
    /// matrix normalized using the [`Sum`](Normalize::normalize_sum) method.
    ///
    /// $$ P_i = \sum_{j=1}^n r_{ij}{w_j} $$
    ///
    /// where $r_{ij}$ is the $i$th alternative (row) and $j$th criterion (column) with $n$ total
    /// criteria, and $w_j$ is the weight of the $j$th criterion.
    ///
    /// Alternatives with a higher $P_i$ value are more preferred.
    ///
    /// # Arguments
    ///
    /// * `weights` - A vector of weights representing the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A vector of preference values if successful.
    ///
    /// # Errors
    ///
    /// * [`RankingError::DimensionMismatch`] - If the number of weights does not match the number
    ///   of criteria.
    /// * [`RankingError::EmptyMatrix`] - If the decision matrix is empty.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::{CriteriaTypes, Rank};
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
    /// let weights = dvector![0.6, 0.4];
    /// let ranking = matrix.rank_weighted_sum(&weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.44, 0.5, 0.58], epsilon = 1e-5);
    /// ```
    fn rank_weighted_sum(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError>;
}

impl Rank for DMatrix<f64> {
    fn rank_aras(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if criteria_types.len() != num_criteria || weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let mut exmatrix = DMatrix::zeros(num_alternatives + 1, num_criteria);
        exmatrix.rows_mut(1, num_alternatives).copy_from(self);

        for (i, criteria_type) in criteria_types.iter().enumerate() {
            exmatrix[(0, i)] = match criteria_type {
                CriterionType::Profit => self.column(i).max(),
                CriterionType::Cost => self.column(i).min(),
            };
        }

        let normalized_matrix = exmatrix.normalize_sum(criteria_types)?;
        let weighted_matrix = normalized_matrix.apply_column_weights(weights)?;

        let s = weighted_matrix.column_sum();
        let k = s
            .rows_range(1..)
            .component_div(&DVector::from_element(num_alternatives, s[0]));

        Ok(k)
    }

    fn rank_cocoso(
        &self,
        weights: &DVector<f64>,
        lambda: Option<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let num_criteria = self.ncols();

        if weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let l = match lambda {
            Some(l) => {
                if (0.0..=1.0).contains(&l) {
                    l
                } else {
                    return Err(RankingError::InvalidValue);
                }
            }
            None => 0.5,
        };

        // Vector of S: sum of weighted rows
        let s = self
            .row_iter()
            .map(|row| row.dot(&weights.transpose()))
            .collect::<Vec<f64>>();
        let s = DVector::from_vec(s);

        // Vector of P: product of rows raised to the power of weights
        let p = self
            .row_iter()
            .map(|row| {
                row.iter()
                    .zip(weights.iter())
                    .map(|(&x, &w)| x.powf(w))
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();
        let p = DVector::from_vec(p);

        // Calculate score strategies
        let s_min = s.min();
        let p_min = p.min();
        let s_max = s.max();
        let p_max = p.max();

        let ksi_a = (&p + &s) / (&p + &s).sum();
        let ksi_b = &s / s_min + &p / p_min;
        let ksi_c = (l * &s + (1.0 - l) * &p) / (l * s_max + (1.0 - l) * p_max);

        // Compute the performance score
        let ksi = (ksi_a.component_mul(&ksi_b).component_mul(&ksi_c)).map(|x| x.powf(1.0 / 3.0))
            + 1.0 / 3.0 * (&ksi_a + &ksi_b + &ksi_c);

        Ok(ksi)
    }

    fn rank_codas(&self, weights: &DVector<f64>, tau: f64) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let weighted_matrix = self.apply_column_weights(weights)?;

        // Compute the Negative Ideal Solution (NIS)
        let nis = weighted_matrix
            .column_iter()
            .map(|col| col.min())
            .collect::<Vec<f64>>();
        let nis = DVector::from_vec(nis).transpose();

        let euclidean_distances = weighted_matrix
            .row_iter()
            .map(|row| {
                row.iter()
                    .zip(nis.iter())
                    .map(|(&x, &ni)| (x - ni).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .collect::<Vec<f64>>();
        let taxicab_distances = weighted_matrix
            .row_iter()
            .map(|row| {
                row.iter()
                    .zip(nis.iter())
                    .map(|(&x, &ni)| (x - ni).abs())
                    .sum::<f64>()
            })
            .collect::<Vec<f64>>();

        let euclidean_distances = DVector::from_vec(euclidean_distances);
        let taxicab_distances = DVector::from_vec(taxicab_distances);

        let mut assessment_matrix = DMatrix::zeros(num_alternatives, num_alternatives);

        for (i, mut row) in assessment_matrix.row_iter_mut().enumerate() {
            for (j, elem) in row.iter_mut().enumerate() {
                let e_diff = euclidean_distances[i] - euclidean_distances[j];
                let t_diff = taxicab_distances[i] - taxicab_distances[j];
                *elem = (e_diff) + (psi(e_diff, tau) * t_diff);
            }
        }

        Ok(assessment_matrix.column_sum())
    }

    fn rank_copras(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if criteria_types.len() != num_criteria || weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let normalized_matrix =
            self.normalize_sum(&CriteriaTypes::all_profits(criteria_types.len()))?;

        let weighted_matrix = normalized_matrix.apply_column_weights(weights)?;

        let sum_normalized_profit = DVector::from_iterator(
            num_alternatives,
            weighted_matrix.row_iter().map(|row| {
                row.iter()
                    .zip(criteria_types.iter())
                    .filter(|&(_, c_type)| *c_type == CriterionType::Profit)
                    .map(|(&val, _)| val)
                    .sum::<f64>()
            }),
        );

        let sum_normalized_cost = DVector::from_iterator(
            num_alternatives,
            weighted_matrix.row_iter().map(|row| {
                row.iter()
                    .zip(criteria_types.iter())
                    .filter(|&(_, c_type)| *c_type == CriterionType::Cost)
                    .map(|(&val, _)| val)
                    .sum::<f64>()
            }),
        );

        let min_sm = sum_normalized_cost.min();

        let q = DVector::from_iterator(
            num_alternatives,
            sum_normalized_profit
                .iter()
                .zip(sum_normalized_cost.iter())
                .map(|(&sp_i, &sm_i)| sp_i + ((min_sm * sm_i) / (sm_i * (min_sm / sm_i)))),
        );

        let max_q = q.max();

        Ok(&q / max_q)
    }

    fn rank_edas(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if criteria_types.len() != num_criteria || weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let average_criteria = self.row_mean();

        let mut positive_distance_matrix = DMatrix::zeros(num_alternatives, num_criteria);
        let mut negative_distance_matrix = DMatrix::zeros(num_alternatives, num_criteria);

        for (j, col) in self.column_iter().enumerate() {
            let avg = average_criteria[j];

            for (i, val) in col.iter().enumerate() {
                match criteria_types[j] {
                    CriterionType::Profit => {
                        positive_distance_matrix[(i, j)] = (val - avg) / avg;
                        negative_distance_matrix[(i, j)] = (avg - val) / avg;
                    }
                    CriterionType::Cost => {
                        positive_distance_matrix[(i, j)] = (avg - val) / avg;
                        negative_distance_matrix[(i, j)] = (val - avg) / avg;
                    }
                }
            }
        }

        // Apply non-negative transformations
        positive_distance_matrix
            .iter_mut()
            .for_each(|x| *x = x.max(0.0));
        negative_distance_matrix
            .iter_mut()
            .for_each(|x| *x = x.max(0.0));

        let sp = positive_distance_matrix
            .apply_column_weights(weights)?
            .column_sum();
        let sn = negative_distance_matrix
            .apply_column_weights(weights)?
            .column_sum();

        let max_sp = sp.max();
        let max_sn = sn.max();

        let nsp = sp / max_sp;
        let nsn = DVector::from_element(num_alternatives, 1.0) - (sn / max_sn);

        Ok((nsp + nsn) / 2.0)
    }

    fn rank_ervd(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
        reference_point: &DVector<f64>,
        lambda: Option<f64>,
        alpha: Option<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        if criteria_types.len() != self.ncols() || weights.len() != self.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        let lambda = lambda.unwrap_or(2.25);
        let alpha = alpha.unwrap_or(0.88);

        let criteria_profits = CriteriaTypes::all_profits(criteria_types.len());
        let normalized_matrix = self.normalize_sum(&criteria_profits)?;
        let reference_point = reference_point.component_div(&self.row_sum().transpose());

        // Calculate the value matrix based on criteria
        let mut value_matrix = normalized_matrix.clone();

        for (i, row) in normalized_matrix.row_iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                value_matrix[(i, j)] = match criteria_types[j] {
                    CriterionType::Profit => {
                        if *value > reference_point[j] {
                            (value - reference_point[j]).powf(alpha)
                        } else {
                            (-1.0 * lambda) * (reference_point[j] - value).powf(alpha)
                        }
                    }
                    CriterionType::Cost => {
                        if *value < reference_point[j] {
                            (reference_point[j] - value).powf(alpha)
                        } else {
                            (-1.0 * lambda) * (value - reference_point[j]).powf(alpha)
                        }
                    }
                }
            }
        }

        // Compute the Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS)
        let pis = value_matrix
            .column_iter()
            .map(|col| col.max())
            .collect::<Vec<f64>>();
        let nis = value_matrix
            .column_iter()
            .map(|col| col.min())
            .collect::<Vec<f64>>();

        let pis = DVector::from_vec(pis).transpose();
        let nis = DVector::from_vec(nis).transpose();

        // Calculate separation measures
        let mut s_plus: DVector<f64> = DVector::zeros(value_matrix.nrows());
        let mut s_minus: DVector<f64> = DVector::zeros(value_matrix.nrows());

        for (i, row) in value_matrix.row_iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                s_plus[i] += weights[j] * (value - pis[j]).abs();
                s_minus[i] += weights[j] * (value - nis[j]).abs();
            }
        }

        let ranking = s_minus.clone().component_div(&(s_plus + s_minus));

        Ok(ranking)
    }

    fn rank_mabac(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        // Calculation of the elements from the weighted matrix
        let weighted_matrix = self.map(|x| x + 1.0).apply_column_weights(weights)?;

        // Border approximation area matrix
        let g = weighted_matrix
            .column_iter()
            .map(|col| {
                col.iter()
                    .product::<f64>()
                    .powf(1.0 / num_alternatives as f64)
            })
            .collect::<Vec<f64>>();

        let g = DVector::from_column_slice(&g).transpose();

        // Distance border approximation area
        let mut q = DMatrix::zeros(num_alternatives, num_criteria);
        for (i, row) in weighted_matrix.row_iter().enumerate() {
            for (j, val) in row.iter().enumerate() {
                q[(i, j)] = val - g[j];
            }
        }

        let ranking = q.row_iter().map(|row| row.sum()).collect::<Vec<f64>>();

        Ok(DVector::from(ranking))
    }

    fn rank_mairca(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        // Theoretical ranking matrix
        let tp = weights / num_alternatives as f64;

        // Real rating matrix
        let tr = self.apply_column_weights(&tp)?;

        // Total gap matrix
        let mut g = tr.clone();

        for (i, row) in tr.row_iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                g[(i, j)] = tp[j] - value;
            }
        }

        Ok(g.column_sum())
    }

    fn rank_marcos(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if criteria_types.len() != num_criteria || weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let max_criteria_values = self.column_iter().map(|x| x.max()).collect::<Vec<f64>>();
        let min_criteria_values = self.column_iter().map(|x| x.min()).collect::<Vec<f64>>();
        let max_criteria_values = DVector::from_vec(max_criteria_values);
        let min_criteria_values = DVector::from_vec(min_criteria_values);

        let mut exmatrix = DMatrix::zeros(num_alternatives + 2, num_criteria);
        exmatrix.rows_mut(0, num_alternatives).copy_from(self);

        for (j, criteria_type) in criteria_types.iter().enumerate() {
            match criteria_type {
                CriterionType::Profit => {
                    exmatrix[(num_alternatives, j)] = max_criteria_values[j];
                    exmatrix[(num_alternatives + 1, j)] = min_criteria_values[j];
                }
                CriterionType::Cost => {
                    exmatrix[(num_alternatives, j)] = min_criteria_values[j];
                    exmatrix[(num_alternatives + 1, j)] = max_criteria_values[j];
                }
            }
        }

        let normalized_exmatrix = exmatrix.normalize_marcos(criteria_types)?;
        let weighted_matrix = normalized_exmatrix.apply_column_weights(weights)?;

        // Utility degree
        let s = weighted_matrix.column_sum();
        let mut k_neg = DVector::zeros(num_alternatives);
        let mut k_pos = DVector::zeros(num_alternatives);
        for i in 0..num_alternatives {
            k_neg[i] = s[i] / s[exmatrix.nrows() - 1];
            k_pos[i] = s[i] / s[exmatrix.nrows() - 2];
        }

        // Utility function
        let sum_k = &k_pos + &k_neg;
        let f_k_pos = k_neg.component_div(&sum_k);
        let f_k_neg = k_pos.component_div(&sum_k);
        let one = DVector::from_element(num_alternatives, 1.0);
        let denominator = &one
            + (&one - &f_k_pos).component_div(&f_k_pos)
            + (&one - &f_k_neg).component_div(&f_k_neg);

        Ok(sum_k.component_div(&denominator))
    }

    fn rank_moora(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if criteria_types.len() != num_criteria || weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let normalized_matrix =
            self.normalize_vector(&CriteriaTypes::all_profits(criteria_types.len()))?;
        let weighted_matrix = normalized_matrix.apply_column_weights(weights)?;

        let sum_normalized_profit = DVector::from_iterator(
            num_alternatives,
            weighted_matrix.row_iter().map(|row| {
                row.iter()
                    .zip(criteria_types.iter())
                    .filter(|&(_, c_type)| *c_type == CriterionType::Profit)
                    .map(|(&val, _)| val)
                    .sum::<f64>()
            }),
        );

        let sum_normalized_cost = DVector::from_iterator(
            num_alternatives,
            weighted_matrix.row_iter().map(|row| {
                row.iter()
                    .zip(criteria_types.iter())
                    .filter(|&(_, c_type)| *c_type == CriterionType::Cost)
                    .map(|(&val, _)| val)
                    .sum::<f64>()
            }),
        );

        let ranking = &sum_normalized_profit - &sum_normalized_cost;

        Ok(ranking)
    }

    fn rank_ocra(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if criteria_types.len() != num_criteria || weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let normalized_matrix = self.normalize_ocra(criteria_types)?;

        let mut i = DVector::zeros(num_alternatives);
        let mut o = DVector::zeros(num_alternatives);
        for (j, col) in normalized_matrix.column_iter().enumerate() {
            match criteria_types[j] {
                CriterionType::Profit => {
                    i += weights[j] * col;
                }
                CriterionType::Cost => {
                    o += weights[j] * col;
                }
            }
        }

        let i_min = i.min();
        let o_min = o.min();
        i.apply(|x| *x -= i_min);
        o.apply(|x| *x -= o_min);

        let i_o_sum = i + o;
        let i_o_min = i_o_sum.min();
        let ranking = i_o_sum.map(|x| x - i_o_min);

        Ok(ranking)
    }

    fn rank_probid(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
        simpler_probid: bool,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if criteria_types.len() != num_criteria || weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let normalized_matrix = self.normalize_vector(&CriteriaTypes::all_profits(num_criteria))?;
        let weighted_matrix = normalized_matrix.apply_column_weights(weights)?;

        let mut pis_matrix = weighted_matrix.clone();
        for (j, column) in weighted_matrix.column_iter().enumerate() {
            let mut column_vec: Vec<f64> = column.iter().copied().collect();
            match criteria_types[j] {
                CriterionType::Profit => {
                    column_vec
                        .sort_by(|a, b| b.partial_cmp(a).unwrap_or(core::cmp::Ordering::Equal));
                    // Descending order
                }
                CriterionType::Cost => {
                    column_vec
                        .sort_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
                    // Ascending order
                }
            }

            pis_matrix
                .column_mut(j)
                .copy_from(&DVector::from_vec(column_vec));
        }

        let average_pis = pis_matrix.row_mean();

        let si = DMatrix::from_fn(num_alternatives, num_alternatives, |i, j| {
            (weighted_matrix.row(i) - pis_matrix.row(j))
                .map(|x| x.powi(2))
                .sum()
                .sqrt()
        });

        let si_average = DVector::from_iterator(
            num_alternatives,
            weighted_matrix
                .row_iter()
                .map(|row| (row - average_pis.clone()).map(|x| x.powi(2)).sum().sqrt()),
        );

        let m = num_alternatives;
        let mut si_pos_ideal = DVector::zeros(m);
        let mut si_neg_ideal = DVector::zeros(m);

        let ranking: DVector<f64> = if simpler_probid {
            if m >= 4 {
                for k in 1..=m / 4 {
                    let factor = 1.0 / k as f64;
                    si_pos_ideal += si.column(k - 1) * factor;
                }

                for k in (m + 1 - m / 4)..=m {
                    let factor = 1.0 / (m - k + 1) as f64;
                    si_neg_ideal += si.column(k - 1) * factor;
                }
            } else {
                si_pos_ideal = si.row(0).transpose();
                si_neg_ideal = si.row(m - 1).transpose();
            }

            si_neg_ideal.component_div(&si_pos_ideal)
        } else {
            let lim = if m % 2 == 1 { (m + 1) / 2 } else { m / 2 };

            for k in 1..=lim {
                let factor = 1.0 / k as f64;
                si_pos_ideal += si.column(k - 1) * factor;
            }

            for k in lim..=m {
                let factor = 1.0 / (m - k + 1) as f64;
                si_neg_ideal += si.column(k - 1) * factor;
            }

            let ri = si_pos_ideal.component_div(&si_neg_ideal);
            DVector::from_iterator(
                m,
                ri.iter()
                    .zip(si_average.iter())
                    .map(|(r, s)| 1.0 / (1.0 + r.powi(2)) + s),
            )
        };

        Ok(ranking)
    }

    fn rank_ram(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if criteria_types.len() != num_criteria || weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let normalized_matrix = &self.normalize_sum(&CriteriaTypes::all_profits(num_criteria))?;
        let weighted_matrix = normalized_matrix.apply_column_weights(weights)?;

        // Vectors of PIS and NIS
        let spi = DVector::from_iterator(
            num_alternatives,
            weighted_matrix.row_iter().map(|row| {
                row.iter()
                    .zip(criteria_types.iter())
                    .filter(|&(_, c_type)| *c_type == CriterionType::Profit)
                    .map(|(&val, _)| val)
                    .sum::<f64>()
            }),
        );
        let smi = DVector::from_iterator(
            num_alternatives,
            weighted_matrix.row_iter().map(|row| {
                row.iter()
                    .zip(criteria_types.iter())
                    .filter(|&(_, c_type)| *c_type == CriterionType::Cost)
                    .map(|(&val, _)| val)
                    .sum()
            }),
        );

        let spi_plus_2 = spi.add_scalar(2.0);
        let smi_plus_2 = smi.add_scalar(2.0);
        let ranking = spi_plus_2.zip_map(&smi_plus_2, |spi_elem, smi_elem| {
            spi_elem.powf(1.0 / smi_elem)
        });

        Ok(ranking)
    }

    fn rank_rim(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
        criteria_range: &DMatrix<f64>,
        reference_ideal: &DMatrix<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if (criteria_types.len() != num_criteria)
            || (weights.len() != num_criteria)
            || (criteria_range.nrows() != num_criteria)
            || (reference_ideal.nrows() != num_criteria)
        {
            return Err(RankingError::DimensionMismatch);
        }

        reference_ideal.is_reference_ideal_bounds_valid(criteria_range)?;

        let normalized_matrix = self.normalize_rim(criteria_range, reference_ideal)?;
        let weighted_normalized_matrix = normalized_matrix.apply_column_weights(weights)?;

        let mut variation_to_pis = DVector::zeros(num_alternatives);
        let mut variation_to_nis = DVector::zeros(num_alternatives);

        for (i, row) in weighted_normalized_matrix.row_iter().enumerate() {
            variation_to_pis[i] = (row - weights.transpose()).map(|x| x.powi(2)).sum().sqrt();
            variation_to_nis[i] = row.map(|x| x.powi(2)).sum().sqrt();
        }

        let variation_to_ideal = DVector::from_vec(
            variation_to_nis
                .iter()
                .zip(&variation_to_pis)
                .map(|(&dm_val, &dp_val)| dm_val / (dm_val + dp_val))
                .collect::<Vec<f64>>(),
        );

        Ok(variation_to_ideal)
    }

    fn rank_spotis(
        &self,
        criteria_types: &CriteriaTypes,
        weights: &DVector<f64>,
        bounds: &DMatrix<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let num_criteria = self.ncols();

        if (criteria_types.len() != num_criteria)
            || (weights.len() != num_criteria)
            || (bounds.nrows() != num_criteria)
        {
            return Err(RankingError::DimensionMismatch);
        }

        bounds.is_bounds_valid()?;
        self.is_within_bounds(bounds)?;

        let normalized_matrix = self.normalize_spotis(criteria_types, bounds)?;

        let ranking = normalized_matrix
            .apply_column_weights(weights)?
            .column_sum();

        Ok(ranking)
    }

    fn rank_topsis(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        let (num_alternatives, num_criteria) = self.shape();

        if weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        if weights.iter().any(|x| *x == 0.0) {
            return Err(RankingError::InvalidValue);
        }

        let broadcasted_weights =
            DMatrix::from_fn(num_alternatives, num_criteria, |_, col| weights[col]);
        let weighted_matrix = self.component_mul(&broadcasted_weights);

        // Compute the Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS)
        let pis = weighted_matrix
            .column_iter()
            .map(|col| col.max())
            .collect::<Vec<f64>>();
        let nis = weighted_matrix
            .column_iter()
            .map(|col| col.min())
            .collect::<Vec<f64>>();

        let pis = DVector::from_vec(pis).transpose();
        let nis = DVector::from_vec(nis).transpose();

        // Calculate the distance to PIS (Dp) and NIS (Dm)
        let mut distance_to_pis = DVector::zeros(num_alternatives);
        let mut distance_to_nis = DVector::zeros(num_alternatives);

        for (i, row) in weighted_matrix.row_iter().enumerate() {
            let dp = (row - &pis).map(|x| x.powi(2)).sum().sqrt();
            distance_to_pis[i] = dp;

            let dn = (row - &nis).map(|x| x.powi(2)).sum().sqrt();
            distance_to_nis[i] = dn;
        }

        let closeness_to_ideal = DVector::from_vec(
            distance_to_pis
                .iter()
                .zip(&distance_to_nis)
                .map(|(&dm_val, &dp_val)| dm_val / (dm_val + dp_val))
                .collect::<Vec<f64>>(),
        );

        Ok(closeness_to_ideal)
    }

    fn rank_waspas(
        &self,
        weights: &DVector<f64>,
        lambda: Option<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        if weights.len() != self.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        let lambda = match lambda {
            Some(l) => {
                if (0.0..=1.0).contains(&l) {
                    l
                } else {
                    return Err(RankingError::InvalidValue);
                }
            }
            None => 0.5,
        };

        let q_sum = self.rank_weighted_sum(weights)?;
        let q_product = self.rank_weighted_product(weights)?;

        let ranking = (lambda * q_sum) + ((1.0 - lambda) * q_product);

        Ok(ranking)
    }

    fn rank_weighted_product(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        if weights.len() != self.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        // Compute the weighted matrix by raising each element of the decision matrix to the power
        // of the corresponding weight.
        let mut weighted_matrix = DMatrix::zeros(self.nrows(), self.ncols());

        // NOTE: I'm sure there is an idiomatic way to do this, but I can't seem to figure it out.
        for (i, row) in self.row_iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                weighted_matrix[(i, j)] = value.powf(weights[j]);
            }
        }

        // Compute the product of each row
        Ok(weighted_matrix.column_product())
    }

    fn rank_weighted_sum(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
        if self.is_empty() {
            return Err(RankingError::EmptyMatrix);
        }

        if weights.len() != self.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        let weighted_matrix = self.apply_column_weights(weights)?;
        Ok(weighted_matrix.column_sum())
    }
}

fn psi(x: f64, tau: f64) -> f64 {
    if x.abs() >= tau {
        1.0
    } else {
        0.0
    }
}
