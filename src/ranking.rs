//! Techniques for ranking alternatives.

use crate::errors::RankingError;
use crate::normalization::Normalize;
use crate::CriteriaType;
use crate::DMatrixExt;
use nalgebra::{DMatrix, DVector};

/// A trait for ranking alternatives in Multiple-Criteria Decision Making (MCDM).
///
/// The [`Rank`] trait defines a method used to rank alternatives based on a normalized or
/// un-normalized decision matrix and a set of weights for the criteria. The ranking process
/// evaluates how well each alternative performs across the criteria, considering the relative
/// importance of each criterion as given by the `weights` array.
///
/// Higher preference values indicate better alternatives. The specific ranking method used (such as
/// [`TOPSIS`](crate::ranking::Rank::rank_topsis) or others) will depend on the implementation of this trait.
///
/// # Example
///
/// Here’s an example of ranking alternatives using the [`Rank`] trait:
///
/// ```rust
/// use mcdm::ranking::Rank;
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
    /// The ARAS method expects the decision matrix before any normalization or manipulation. The method
    /// assesses alternatives by comparing their overall performance to the ideal (best) alternative. It
    /// calculates a utility degree for each alternative based on the ratio of the sum of weighted
    /// normalized values for each criterion relative to the ideal alternative, which has the maximum
    /// performance for each criterion.
    ///
    /// This method takes an $n{\times}m$ decision matrix
    ///
    /// $$ x_{ij} =
    /// \begin{bmatrix}
    /// x_{11} & x_{12} & \ldots & x_{1m} \\\\
    /// x_{21} & x_{22} & \ldots & x_{2m} \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// x_{n1} & x_{n2} & \ldots & x_{nm}
    /// \end{bmatrix}
    /// $$
    ///
    /// then extends the matrix by adding an additional "best case" alternative row based on the
    /// minimum or maximum values of each criterion column. If that criterion is a profit, we use the
    /// maximum; if the criterion is a cost, we use the minimum.
    ///
    /// $$ E =
    /// \begin{bmatrix}
    ///     E_0(x_{i1}) & E_0(x_{i2}) & \ldots & E_0(x_{im}) \\\\
    ///     x_{11} & x_{12} & \ldots & x_{1m} \\\\
    ///     x_{21} & x_{22} & \ldots & x_{2m} \\\\
    ///     \vdots & \vdots & \ddots & \vdots \\\\
    ///     x_{(n+1)1} & x_{(n+1)2} & \ldots & x_{(n+1)m}
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
    /// Next, obtain the normalized matrix, $s_{ij}$ by using the [`Sum`](crate::normalization::Normalize::normalize_sum) normalization method on $E$.
    /// Then compute the weighted matrix $v_{ij}$ using
    ///
    /// $$ v_{ij} = w_j s_{ij} $$
    ///
    /// Next, determine the optimal criterion values only for the extended "best case" alternative
    /// (remember, this is the first row of the extended matrix).
    ///
    /// $$ S_0 = \sum_{j=1}^m v_{0j} $$
    ///
    /// Likewise, determine the sum of each other alternative using
    ///
    /// $$ S_i = \sum_{j=1}^m v_{ij} $$
    ///
    /// Lastly, calculate the utility degree $K_i$ which determines the ranking of each alternative
    ///
    /// $$ K_i = \frac{S_i}{S_0} $$
    ///
    /// # Arguments
    ///
    /// * `types` - A 1D array of criterion types.
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_aras(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.49447117, 0.35767527, 1.0], epsilon = 1e-5);
    /// ```
    fn rank_aras(
        &self,
        types: &[CriteriaType],
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the COmbined Compromise SOlution (COCOSO) method.
    ///
    /// The COCOSO method expects the decision matrix is normalized using the [`MinMax`](crate::normalization::Normalize::normalize_min_max)
    /// method. Then calculates the weighted sum of the comparision sequence and the total power weight
    /// of the comparison sequence for each alternative. The values of $S_i$ are based on the grey
    /// relationship generation method and the values for $P_i$ are based on the multiplicative WASPAS
    /// method.
    ///
    /// $$ S_i = \sum_{j=1}^m(w_j r_{ij}) $$
    /// $$ P_i = \sum_{j=1}^m(r_{ij})^{w_j} $$
    ///
    /// where $S_i$ is the grey relationship, $P_i$ is the multiplicative `WASPAS`, $m$ is the number of
    /// criteria, $r_{ij}$ is the $i$th element of the alternative, $j$th elements of the criterion of
    /// the normalized decision matrix, and $w_j$ is the $j$th weight.
    ///
    /// We then compute the relative weights of alternatives using aggregation strategies.
    ///
    /// $$ k_{ia} = \frac{P_i + S_i}{\sum_{i=1}^n \left(P_i + S_i\right)} $$
    /// $$ k_{ib} = \frac{S_i}{\min_i(S_i)} + \frac{P_i}{\min_i(P_i)} $$
    /// $$ k_{ic} = \frac{\lambda(S_i) + (1 - \lambda)(P_i)}{\lambda \max_i(S_i) + (1 - \lambda) \max_i(P_i)} \quad 0 \leq \lambda \leq 1 $$
    ///
    /// where $k_{ia}$ represents the average of the sums of [`WeightedSum`](crate::ranking::Rank::rank_weighted_sum)
    /// and [`WeightedProduct`](crate::ranking::Rank::rank_weighted_product) scores, $k_{ib}$
    /// represents the [`WeightedSum`](crate::ranking::Rank::rank_weighted_sum) and
    /// [`WeightedProduct`](crate::ranking::Rank::rank_weighted_product) scores over the best scores
    /// for each each method respectfully, and $k_{ic}$ represents the [`WeightedSum`](crate::ranking::Rank::rank_weighted_sum)
    /// and [`WeightedProduct`](crate::ranking::Rank::rank_weighted_product) scores using the
    /// compromise strategy, and $n$ is the number of alternatives.
    ///
    /// Lastly, we rank the alternatives as follows:
    ///
    /// $$ k_i = (k_{ia}k_{ib}k_{ic})^{\frac{1}{3}} + \frac{1}{3}(k_{ia} + k_{ib} + k_{ic}) $$
    ///
    /// # Arguments
    ///
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use mcdm::normalization::Normalize;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_min_max(&criteria_types).unwrap();
    /// let ranking = normalized_matrix.rank_cocoso(&weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![3.24754746, 1.14396494, 5.83576765], epsilon = 1e-5);
    /// ```
    fn rank_cocoso(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the COmbinative Distance-based ASessment (CODAS) method.
    ///
    /// The CODAS method expects the decision matrix is normalized using the [`Linear`](crate::normalization::Normalize::normalize_linear)
    /// method. Then calculates an assessment matrix based on the euclidean distance and taxicab
    /// distance from the negative ideal solution.
    ///
    /// Build a weighted matrix $v_{ij}$ using the normalized decision matrix, $r_{ij}$, and weights.
    ///
    /// $$ v_{ij} = r_{ij}{w_j} $$
    ///
    /// Next, determine the negative ideal solution (NIS) using the weighted matrix $v_{ij}$.
    ///
    /// $$ NIS_j = \min_{i=1}^n v_{ij} $$
    ///
    /// Calculate the euclidean distance and taxicab distance from the negative ideal solution
    ///
    /// $$ E_i = \sqrt{\sum_{i=1}^n(v_{ij} - NIS_j)^2} $$
    /// $$ T_i = \sum_{i=1}^n \left|v_{ij} - NIS_j\right| $$
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
    /// $$ H_i = \sum_{k=1}^n h_{ik} $$
    ///
    /// # Arguments
    ///
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    /// * `tau` - The threshold value for the threshold function. Default is 0.02.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use mcdm::normalization::Normalize;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_linear(&criteria_types).unwrap();
    /// let ranking = normalized_matrix.rank_codas(&weights, 0.02).unwrap();
    /// assert_relative_eq!(ranking, dvector![-0.40977725, -1.15891275, 1.56869], epsilon = 1e-5);
    /// ```
    fn rank_codas(&self, weights: &DVector<f64>, tau: f64) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the COmplex PRoportional ASsessment (COPRAS) method.
    ///
    /// The COPRAS method expects the decision matrix without normalization. This method evaluates
    /// alternatives by separately considering the effects of maximizing (beneficial) and minimizing
    /// (non-beneficial) index values of attributes. This approach allows COPRAS to assess the impact of
    /// each type of crition independently, ensuring both positive contributions and cost factors are
    /// accounted for in the final ranking. This separation provides a more balanced and accurate
    /// assessment of each alternative.
    ///
    /// Start by calculating the normalized decision matrix, $r_{ij}$, using the [`Sum`](crate::normalization::Normalize::normalize_sum) method, but treat each
    /// criterion as a profit. The normalization is caclculated as:
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
    /// $$ S_{-i} = \sum_{j=k+1}^m v_{ij} $$
    ///
    /// where $k$ is the number of attributes to maximize. The rest of the attributes from $k+1$ to $m$
    /// are minimized. $S_{+i}$ and $S_{-i}$ show the level of the goal achievement for alternatives.
    /// Higher value of $S_{+i}$ indicates the alternative is better and a lower value of $S_{-i}$
    /// indicate a better alternative.
    ///
    /// Next, calculate the relative significance of alternatives using:
    ///
    /// $$ Q_i = S_{+i} + \frac{S_{-\min} \sum_{i=1}^n S_{-i}}{S_{-i} \sum_{i=1}^n \left(\frac{S_{-\min}}{S_{-i}}\right)} $$
    ///
    /// Lastly, rank the alternatives using:
    ///
    /// $$ U_i = \frac{Q_i}{Q_i^{\max}} \times 100\\% $$
    ///
    /// where $Q_i^{\max}$ is the maximum value of the utility function. Better alternatives have higher
    /// $U_i$ values.
    ///
    /// # Arguments
    ///
    /// * `types` - A 1D array of criterion types.
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_copras(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![1.0, 0.6266752, 0.92104753], epsilon = 1e-5);
    /// ```
    fn rank_copras(
        &self,
        types: &[CriteriaType],
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the Evaluation based on Distance from Average Solution (EDAS) method.
    ///
    /// The EDAS method ranks the alternatives using the average distance from the average solution. The
    /// method expects a decision matrix before normalization. We define the decision matrix as:
    ///
    /// $$ X_{ij} =
    /// \begin{bmatrix}
    ///     x_{11} & x_{12} & \ldots & x_{1m} \\\\
    ///     x_{21} & x_{22} & \ldots & x_{2m} \\\\
    ///     \vdots & \vdots & \ddots & \vdots \\\\
    ///     x_{n1} & x_{n2} & \ldots & x_{nm}
    /// \end{bmatrix}
    /// $$
    ///
    /// Then calculate the average solution as:
    ///
    /// $$ \overline{X}\_{ij} = \frac{\sum_{i=1}^{n} x_{ij}}{n} $$
    ///
    /// Next, calculate the positive and negative distance from the mean solution for each alternative.
    /// When the criteria type is profit, compute the positive and negative distance as:
    ///
    /// $$ PD_{i} = \frac{\max(0, (X_{ij} - \overline{X}\_{ij}))}{\overline{X}\_{ij}} $$
    /// $$ ND_{i} = \frac{\max(0, (\overline{X}\_{ij})) - X_{ij}}{\overline{X}\_{ij}} $$
    ///
    /// When the criter type is cost, compute the positive and negative distance as:
    ///
    /// $$ PD_{i} = \frac{\max(0, (\overline{X}\_{ij})) - X_{ij}}{\overline{X}\_{ij}} $$
    /// $$ ND_{i} = \frac{\max(0, (X_{ij} - \overline{X}\_{ij}))}{\overline{X}\_{ij}} $$
    ///
    /// Next, calculate the weighted sums for $PD$ and $ND$:
    ///
    /// $$ SP_i = \sum_{j=1}^{m} w_j PD_{ij} $$
    /// $$ SN_i = \sum_{j=1}^{m} w_j ND_{ij} $$
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
    /// # Arguments
    ///
    /// * `types` - A 1D array of criterion types.
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_edas(&criteria_types, &weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.04747397, 0.04029913, 1.0], epsilon = 1e-5);
    /// ```
    fn rank_edas(
        &self,
        types: &[CriteriaType],
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the Enhanced Relative Value Decomposition (ERVD) method.
    ///
    /// The ERVD method expects the decision matrix is not normalized.
    ///
    /// To evaluate decision alternatives, start by defining a decision matrix, $d_{ij}$, where $i$
    /// represents the alternatives and $j$ represents the criteria.
    ///
    /// Then, define the reference points $\mu,j=1,\ldots,m$ for each decision criterion.
    ///
    /// Next, normalize the decision matrix using the [`Sum`](crate::normalization::Normalize::normalize_sum)
    /// method, which gives us the normalized decision matrix $r_{ij}$.
    ///
    /// Next, transform the reference points into the normalized scale:
    ///
    /// $$ \varphi_j = \frac{\mu_j}{\sum_{i=1}^n d_{ij}} $$
    ///
    /// where $\mu_j$ is the $j$th element of the reference point vector and $d_{ij}$ is the $i$th
    /// row and $j$th column of the decision matrix.
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
    /// $$ A^+ = \left\\{ v_1^+, \ldots, v_m^+ \right\\} $$
    /// $$ A^- = \left\\{ v_1^-, \ldots, v_m^- \right\\} $$
    ///
    /// where $v_j^+ = \max_i(v_{ij})$ and $v_j^- = \min_i(v_{ij})$.
    ///
    /// Next, calculate the separation measures, $S_i$, from PIS and NIS individually using the
    /// [Minkowski metric](https://en.wikipedia.org/wiki/Minkowski_distance):
    ///
    /// $$ S_i^+ = \sum_{j=1}^m w_j \cdot \left| v_{ij} - v_j^+ \right| $$
    /// $$ S_i^- = \sum_{j=1}^m w_j \cdot \left| v_{ij} - v_j^- \right| $$
    ///
    /// Finally, rank the alternatives by calculating their relative closeness to the ideal solution:
    ///
    /// $$ \phi_i = \frac{S_i^-}{S_i^+ + S_i^-} \quad \text{for} \quad i=1, \ldots, n$$
    ///
    /// # Arguments
    ///
    /// * `types` - A 1D array of criterion types.
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    /// * `reference_point` - A 1D array of reference points corresponding to the relative importance
    ///   of each criterion.
    /// * `lambda` - A scalar value representing attenuation factor for the losses. Suggested to be
    ///   between 2.0 and 2.5.Default is 2.25.
    /// * `alpha` - A scalar value representing the diminishing sensitivity parameter. Default is
    ///   0.88.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let reference_point = dvector![1.46666667, 2.04333333, 0.84, 2.02];
    /// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
    /// let ranking = matrix.rank_ervd(&criteria_types, &weights, &reference_point, 0.5, 0.5).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.30321682, 0.216203469, 1.0], epsilon = 1e-5);
    /// ```
    fn rank_ervd(
        &self,
        types: &[CriteriaType],
        weights: &DVector<f64>,
        reference_point: &DVector<f64>,
        lambda: f64,
        alpha: f64,
    ) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the Multi-Attributive Border Approximation Area Comparison (MABAC)
    /// method.
    ///
    /// The MABAC method expects the decision matrix is normalized using the [`MinMax`](crate::normalization::Normalize::normalize_min_max)
    /// method. Then computes a weighted matrix $v_{ij}$ using
    ///
    /// $$ v_{ij} = {w_j}(x_{ij} + 1) $$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column), and $w_j$ is the weight of the $j$th criterion.
    ///
    /// We then compute the boundary appromixation area for all criteria.
    ///
    /// $$ g_i = \left( \prod_{j=1}^m v_{ij} \right)^{1/m} $$
    ///
    /// where $g_i$ is the boundary approximation area for the $i$th alternative, $v_{ij}$ is the
    /// weighted matrix for the $i$th alternative and $j$th criterion, and $m$ is the number of
    /// criteria.
    ///
    /// Next we calculate the distance of the $i$th alternative and $j$th criterion from the boundary
    /// approximation area
    ///
    /// $$ q_{ij} = v_{ij} - g_j $$
    ///
    /// Lastly, we rank the alternatives according to the sum of the distances of the alternatives from
    /// the border approximation area.
    ///
    /// $$ S_i = \sum_{j=1}^{m} q_{ij} \quad \text{for} \quad i=1, \ldots, n \quad \text{and} \quad j=1, \ldots, m $$
    ///
    /// where $q_{ij}$ is the distance of the $i$th alternative and $j$th criterion of the weighted
    /// matrix $v_{ij}$ to the boundary approximation $g_i$, $n$ is the number of alternatives and $m$
    /// is the number of criteria.
    ///
    /// # Arguments
    ///
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use mcdm::normalization::Normalize;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_min_max(&criteria_types).unwrap();
    /// let ranking = normalized_matrix.rank_mabac(&weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![-0.01955314, -0.31233795,  0.52420052], epsilon = 1e-5);
    /// ```
    fn rank_mabac(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError>;

    /// Ranks alternatives using the Multi-Attributive Ideal-Real Comparative Analysis (MAIRCA)
    /// method.
    ///
    /// The MAIRCA method operates on a normalized decision matrix. The typical normalization method
    /// used is the [`MinMax`](crate::normalization::Normalize::normalize_min_max) method.
    ///
    /// To start, we define a normalied $n \times m$ decision matrix where $n$ is the number of
    /// alternatives and $m$ is the number of criteria.
    ///
    /// $$ x_{ij} =
    /// \begin{bmatrix}
    /// x_{11} & x_{12} & \ldots & x_{1m} \\\\
    /// x_{21} & x_{22} & \ldots & x_{2m} \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// x_{n1} & x_{n2} & \ldots & x_{nm}
    /// \end{bmatrix}
    /// $$
    ///
    /// Next, calculate the preference for choosing alternatives using the vector $P_{Ai}$ where
    ///
    /// $$ P_{Ai} = \frac{1}{n} $$
    ///
    /// Next, calculate a theoretical ranking matrix $T_p$ where
    ///
    /// $$ T_p =
    /// \begin{bmatrix}
    /// t_{p11} & t_{p12} & \ldots & t_{p1m} \\\\
    /// t_{p21} & t_{p22} & \ldots & t_{p2m} \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// t_{pn1} & t_{pn2} & \ldots & t_{pnm}
    /// \end{bmatrix} =
    /// \begin{bmatrix}
    /// P_{A1} \cdot w_1 & P_{A1} \cdot w_2 & \ldots & P_{A1} \cdot w_m \\\\
    /// P_{A2} \cdot w_1 & P_{A2} \cdot w_2 & \ldots & P_{A2} \cdot w_m \\\\
    /// \vdots           & \vdots           & \ddots & \vdots \\\\
    /// P_{An} \cdot w_1 & P_{An} \cdot w_2 & \ldots & P_{An} \cdot w_m
    /// \end{bmatrix}
    /// $$
    ///
    /// Next, calculate the real rating matrix
    ///
    /// $$ T_r =
    /// \begin{bmatrix}
    /// t_{r11} & t_{r12} & \ldots & t_{r1m} \\\\
    /// t_{r21} & t_{r22} & \ldots & t_{r2m} \\\\
    /// \vdots & \vdots & \ddots & \vdots \\\\
    /// t_{rn1} & t_{rn2} & \ldots & t_{rnm}
    /// \end{bmatrix}
    /// $$
    ///
    /// The values of the real rating matrix are dependent on the criteria type. If the criteria
    /// type is profit:
    ///
    /// $$ t_{rij} = t_{pij} \cdot \left(  \frac{x_{ij} - \min(x_j)}{\max(x_j) - \min(x_j)} \right) $$
    ///
    /// if the criteria type is cost:
    ///
    /// $$ t_{rij} = t_{pij} \cdot \left(  \frac{x_{ij} - \max(x_j)}{\min(x_j) - \max(x_j)} \right) $$
    ///
    /// Next, calculate the total gap matrix, $G$, by taking the element-wise difference between the
    /// theoretical ranking method and the real rating matrix.
    ///
    /// $$ G = T_p - T_r  =
    /// \begin{bmatrix}
    /// t_{p11} - t_{r11} & t_{p12} - t_{r12} & \ldots & t_{p1m} - t_{r1m} \\\\
    /// t_{p21} - t_{r21} & t_{p22} - t_{r22} & \ldots & t_{p2m} - t_{r2m} \\\\
    /// \vdots            & \vdots            & \ddots & \vdots \\\\
    /// t_{pn1} - t_{rn1} & t_{pn2} - t_{rn2} & \ldots & t_{pnm} - t_{rnm}
    /// \end{bmatrix}
    /// $$
    ///
    /// Finally, rank the alternatives using the sum of the rows of the gap matrix, $G$.
    ///
    /// $$ Q_i = \sum_{j=1}^m g_{ij} $$
    ///
    /// Lower values of $Q_i$ indicate that alternative $i$ is ranked higher.
    ///
    /// # Arguments
    ///
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use mcdm::normalization::Normalize;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_min_max(&criteria_types).unwrap();
    /// let ranking = normalized_matrix.rank_mairca(&weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.18125122, 0.27884615, 0.0], epsilon = 1e-5);
    /// ```
    fn rank_mairca(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError>;

    /// Ranks the alternatives using the TOPSIS method.
    ///
    /// The TOPSIS method expects the decision matrix is normalized using the [`MinMax`](crate::normalization::Normalize::normalize_min_max)
    /// method. Then computes a weighted matrix $v_{ij}$ using
    ///
    /// $$ v_{ij} = r_{ij}{w_j} $$
    ///
    /// where $r_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column) of the normalized decision matrix, and $w_j$ is the weight of the $j$th criterion.
    ///
    /// We then derive a positive ideal solution (PIS), $A_j^+$, and a negative ideal solution (NIS),
    /// $A_j^-$.
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
    /// # Arguments
    ///
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use mcdm::normalization::Normalize;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
    /// let normalized_matrix = matrix.normalize_min_max(&criteria_types).unwrap();
    /// let ranking = normalized_matrix.rank_topsis(&weights).unwrap();
    /// assert_relative_eq!(ranking, dvector![0.52910451, 0.72983217, 0.0], epsilon = 1e-5);
    /// ```
    fn rank_topsis(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError>;

    /// Computes the Weighted Product Model (WPM) preference values for alternatives.
    ///
    /// The WPM model expects the decision matrix is normalized using the [`Sum`](crate::normalization::Normalize::normalize_sum) method. Then computes
    /// the ranking using:
    ///
    /// $$ WPM = \prod_{j=1}^n(x_{ij})^{w_j} $$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column) with $n$ total criteria, and $w_j$ is the weight of the $j$th criterion.
    ///
    /// # Arguments
    ///
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
    /// use mcdm::normalization::Normalize;
    /// use mcdm::CriteriaType;
    /// use nalgebra::{dmatrix, dvector};
    ///
    /// let matrix = dmatrix![
    ///     2.9, 2.31, 0.56, 1.89;
    ///     1.2, 1.34, 0.21, 2.48;
    ///     0.3, 2.48, 1.75, 1.69
    /// ];
    /// let weights = dvector![0.25, 0.25, 0.25, 0.25];
    /// let criteria_type = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
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
    /// The `WeightedSum` method ranks alternatives based on the weighted sum of their criteria values.
    /// Each alternative's score is calculated by multiplying its criteria values by the corresponding
    /// weights and summing the results. The decision matrix is expected to be normalized using the
    /// [`Sum`](crate::normalization::Normalize::normalize_sum) method.
    ///
    /// $$ WSM = \sum_{j=1}^n x_{ij}{w_j} $$
    ///
    /// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
    /// (column) with $n$ total criteria, and $w_j$ is the weight of the $j$th criterion.
    ///
    /// # Arguments
    ///
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<DVector<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    ///
    /// # Example
    ///
    /// ```rust
    /// use approx::assert_relative_eq;
    /// use mcdm::ranking::Rank;
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
        types: &[CriteriaType],
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(RankingError::EmptyMatrix);
        }

        if types.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let mut exmatrix = DMatrix::zeros(num_alternatives + 1, num_criteria);
        exmatrix.rows_mut(1, num_alternatives).copy_from(self);

        for (i, criteria_type) in types.iter().enumerate() {
            if *criteria_type == CriteriaType::Profit {
                exmatrix[(0, i)] = self.column(i).max();
            } else if *criteria_type == CriteriaType::Cost {
                exmatrix[(0, i)] = self.column(i).min();
            }
        }

        let normalized_matrix = exmatrix.normalize_sum(types)?;
        let weighted_matrix = normalized_matrix.scale_columns(weights);

        let s = weighted_matrix.column_sum();

        let k = s
            .rows_range(1..)
            .component_div(&DVector::from_element(num_alternatives, s[0]));

        Ok(k)
    }

    fn rank_cocoso(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
        if weights.len() != self.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        let l = 0.5;

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
        if weights.len() != self.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        let weighted_matrix = self.scale_columns(weights);

        let nrows = self.nrows();

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

        let mut assessment_matrix = DMatrix::zeros(nrows, nrows);

        for i in 0..nrows {
            for j in 0..nrows {
                let e_diff = euclidean_distances[i] - euclidean_distances[j];
                let t_diff = taxicab_distances[i] - taxicab_distances[j];
                assessment_matrix[(i, j)] = (e_diff) + (psi(e_diff, tau) * t_diff);
            }
        }

        Ok(assessment_matrix.column_sum())
    }

    fn rank_copras(
        &self,
        types: &[CriteriaType],
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(RankingError::EmptyMatrix);
        }

        if types.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let normalized_matrix = self.normalize_sum(&CriteriaType::profits(types.len()))?;

        let weighted_matrix = normalized_matrix.scale_columns(weights);

        let sum_normalized_profit = DVector::from_iterator(
            num_alternatives,
            weighted_matrix.row_iter().map(|row| {
                row.iter()
                    .zip(types.iter())
                    .filter(|&(_, c_type)| *c_type == CriteriaType::Profit)
                    .map(|(&val, _)| val)
                    .sum::<f64>()
            }),
        );

        let sum_normalized_cost = DVector::from_iterator(
            num_alternatives,
            weighted_matrix.row_iter().map(|row| {
                row.iter()
                    .zip(types.iter())
                    .filter(|&(_, c_type)| *c_type == CriteriaType::Cost)
                    .map(|(&val, _)| val)
                    .sum::<f64>()
            }),
        );

        let min_sm = sum_normalized_cost
            .iter()
            .cloned()
            .reduce(f64::min)
            .unwrap_or(0.0);

        let q = DVector::from_iterator(
            num_alternatives,
            sum_normalized_profit
                .iter()
                .zip(sum_normalized_cost.iter())
                .map(|(&sp_i, &sm_i)| sp_i + ((min_sm * sm_i) / (sm_i * (min_sm / sm_i)))),
        );

        let max_q = q.iter().cloned().reduce(f64::max).unwrap_or(1.0);

        Ok(&q / max_q)
    }

    fn rank_edas(
        &self,
        types: &[CriteriaType],
        weights: &DVector<f64>,
    ) -> Result<DVector<f64>, RankingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(RankingError::EmptyMatrix);
        }

        if types.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let average_criteria = self.row_mean();

        let mut positive_distance_matrix = DMatrix::zeros(num_alternatives, num_criteria);
        let mut negative_distance_matrix = DMatrix::zeros(num_alternatives, num_criteria);

        for j in 0..num_criteria {
            for i in 0..num_alternatives {
                let val = self[(i, j)];
                let avg = average_criteria[j];

                match types[j] {
                    CriteriaType::Profit => {
                        positive_distance_matrix[(i, j)] = (val - avg) / avg;
                        negative_distance_matrix[(i, j)] = (avg - val) / avg;
                    }
                    CriteriaType::Cost => {
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

        let sp = positive_distance_matrix.scale_columns(weights).column_sum();
        let sn = negative_distance_matrix.scale_columns(weights).column_sum();

        let max_sp = sp.max();
        let max_sn = sn.max();

        let nsp = sp / max_sp;
        let nsn = DVector::from_element(num_alternatives, 1.0) - (sn / max_sn);

        Ok((nsp + nsn) / 2.0)
    }

    fn rank_ervd(
        &self,
        types: &[CriteriaType],
        weights: &DVector<f64>,
        reference_point: &DVector<f64>,
        lambda: f64,
        alpha: f64,
    ) -> Result<DVector<f64>, RankingError> {
        let criteria_profits = CriteriaType::profits(types.len());
        let normalized_matrix = self.normalize_sum(&criteria_profits)?;
        let reference_point = reference_point.component_div(&self.row_sum().transpose());

        // Calculate the value matrix based on criteria
        let mut value_matrix = normalized_matrix.clone();

        for (i, row) in normalized_matrix.row_iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                match types[j] {
                    CriteriaType::Profit => {
                        if *value > reference_point[j] {
                            value_matrix[(i, j)] = (value - reference_point[j]).powf(alpha);
                        } else {
                            value_matrix[(i, j)] =
                                (-1.0 * lambda) * (reference_point[j] - value).powf(alpha);
                        }
                    }
                    CriteriaType::Cost => {
                        if *value < reference_point[j] {
                            value_matrix[(i, j)] = (reference_point[j] - value).powf(alpha);
                        } else {
                            value_matrix[(i, j)] =
                                (-1.0 * lambda) * (value - reference_point[j]).powf(alpha);
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

        let mut ranking = DVector::zeros(self.nrows());

        for i in 0..ranking.nrows() {
            ranking[i] = s_minus[i] / (s_plus[i] + s_minus[i]);
        }

        Ok(ranking)
    }

    fn rank_mabac(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
        let (num_alternatives, num_criteria) = self.shape();

        if weights.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        // Calculation of the elements from the weighted matrix
        let weighted_matrix = self.map(|x| x + 1.0).scale_columns(weights);

        // Border approximation area matrix
        let g = weighted_matrix
            .column_iter()
            .map(|col| col.iter().product::<f64>().powf(1.0 / self.nrows() as f64))
            .collect::<Vec<f64>>();

        let g = DVector::from_column_slice(&g).transpose();

        // Distance border approximation area
        let mut q = DMatrix::zeros(num_alternatives, num_criteria);
        for (i, row) in weighted_matrix.row_iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                q[(i, j)] = value - g[j];
            }
        }

        let ranking = q.row_iter().map(|row| row.sum()).collect::<Vec<f64>>();

        Ok(DVector::from(ranking))
    }

    fn rank_mairca(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
        if weights.len() != self.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        // Theoretical ranking matrix
        let tp = weights / self.nrows() as f64;

        // Real rating matrix
        let tr = self.scale_columns(&tp);

        // Total gap matrix
        let mut g = tr.clone();

        for (i, row) in tr.row_iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                g[(i, j)] = tp[j] - value;
            }
        }

        Ok(g.column_sum())
    }

    fn rank_topsis(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
        if weights.len() != self.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        if weights.iter().any(|x| *x == 0.0) {
            return Err(RankingError::InvalidValue);
        }

        let (num_rows, num_cols) = self.shape();

        let broadcasted_weights = DMatrix::from_fn(num_rows, num_cols, |_, col| weights[col]);
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
        let mut distance_to_pis = DVector::zeros(num_rows);
        let mut distance_to_nis = DVector::zeros(num_rows);

        for (n, row) in weighted_matrix.row_iter().enumerate() {
            let dp = (row - &pis).map(|x| x.powi(2)).sum().sqrt();
            distance_to_pis[n] = dp;

            let dn = (row - &nis).map(|x| x.powi(2)).sum().sqrt();
            distance_to_nis[n] = dn;
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

    fn rank_weighted_product(&self, weights: &DVector<f64>) -> Result<DVector<f64>, RankingError> {
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
        if weights.len() != self.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        let weighted_matrix = self.scale_columns(weights);
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
