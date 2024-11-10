//! Techniques for ranking alternatives.

use crate::errors::RankingError;
use crate::normalization::{Normalize, Sum};
use crate::CriteriaType;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_stats::QuantileExt;

/// A trait for ranking alternatives in Multiple-Criteria Decision Making (MCDM).
///
/// The [`Rank`] trait defines a method used to rank alternatives based on a normalized decision
/// matrix and a set of weights for the criteria. The ranking process evaluates how well each
/// alternative performs across the criteria, considering the relative importance of each criterion
/// as given by the `weights` array.
///
/// Higher preference values indicate better alternatives. The specific ranking method used (such as
/// [`TOPSIS`] or others) will depend on the implementation of this trait.
///
/// # Example
///
/// Here’s an example of ranking alternatives using the [`Rank`] trait:
///
/// ```rust
/// use mcdm::ranking::{TOPSIS, Rank};
/// use ndarray::{array, Array1, Array2};
///
/// let normalized_matrix: Array2<f64> = array![[0.8, 0.6], [0.5, 0.9], [0.3, 0.7]];
/// let weights: Array1<f64> = array![0.6, 0.4];
/// let ranking = TOPSIS::rank(&normalized_matrix, &weights).unwrap();
/// println!("Ranking: {:?}", ranking);
/// ```
pub trait Rank {
    /// Ranks the alternatives of a normalized decision matrix based on the provided criteria
    /// weights.
    ///
    /// This method computes preference values for each alternative in the decision matrix by
    /// applying the weights for each criterion. The alternatives are ranked based on these
    /// preference values, with higher values indicating better alternatives.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A normalized decision matrix where each row represents an alternative and each
    ///   column represents a criterion.
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, RankingError>;
}

/// A trait for ranking alternatives in Multiple-Criteria Decision Making (MCDM).
///
/// The [`RankWithCriteriaType`] trait defines a method used to rank alternatives based on a
/// decision matrix and a set of weights for the criteria. The ranking process evaluates how well
/// each alternative performs across the criteria, considering the relative importance of each
/// criterion as given by the `weights` array.
///
/// Higher preference values indicate better alternatives. The specific ranking method used (such as
/// [`Aras`] or others) will depend on the implementation of this trait.
///
/// # Example
///
/// Here’s an example of ranking alternatives using the [`RankWithCriteriaType`] trait:
///
/// ```rust
/// use mcdm::ranking::{Aras, RankWithCriteriaType};
/// use ndarray::{array, Array1, Array2};
///
/// let normalized_matrix: Array2<f64> = array![[0.8, 0.6], [0.5, 0.9], [0.3, 0.7]];
/// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1]).unwrap();
/// let weights: Array1<f64> = array![0.6, 0.4];
/// let ranking = Aras::rank(&normalized_matrix, &criteria_types, &weights).unwrap();
/// println!("Ranking: {:?}", ranking);
/// ```
pub trait RankWithCriteriaType {
    /// Ranks the alternatives of a decision matrix based on the provided criteria types and
    /// weights.
    ///
    /// This method computes preference values for each alternative in the decision matrix by
    /// applying the weights for each criterion and accounting for each criteria being a cost or
    /// profit. The alternatives are ranked based on these preference values, with higher values
    /// indicating better alternatives.
    ///
    /// # Arguments
    ///
    /// * `matrix` - A normalized decision matrix where each row represents an alternative and each
    ///   column represents a criterion.
    /// * `types` - An array of criteria types indicating whether each criterion is a cost or profit.
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each
    ///   criterion.
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    fn rank(
        matrix: &Array2<f64>,
        types: &[CriteriaType],
        weights: &Array1<f64>,
    ) -> Result<Array1<f64>, RankingError>;
}

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
/// Next, obtain the normalized matrix, $s_{ij}$ by using the [`Sum`] normalization method on $E$.
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
/// # Example
///
/// ```rust
/// use approx::assert_abs_diff_eq;
/// use mcdm::ranking::{RankWithCriteriaType, Aras};
/// use mcdm::normalization::{Sum, Normalize};
/// use ndarray::{array, Array2};
///
/// let matrix = array![
///     [2.9, 2.31, 0.56, 1.89],
///     [1.2, 1.34, 0.21, 2.48],
///     [0.3, 2.48, 1.75, 1.69]
/// ];
/// let weights = array![0.25, 0.25, 0.25, 0.25];
/// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
/// let normalized_matrix = Sum::normalize(&matrix, &criteria_types).unwrap();
/// let ranking = Aras::rank(&matrix, &criteria_types, &weights).unwrap();
/// assert_abs_diff_eq!(ranking, array![0.49447117, 0.35767527, 1.0], epsilon = 1e-5);
/// ```
pub struct Aras;

impl RankWithCriteriaType for Aras {
    fn rank(
        matrix: &Array2<f64>,
        types: &[CriteriaType],
        weights: &Array1<f64>,
    ) -> Result<Array1<f64>, RankingError> {
        let (num_alternatives, num_criteria) = matrix.dim();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(RankingError::EmptyMatrix);
        }

        if types.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let mut exmatrix = Array2::<f64>::zeros((num_alternatives + 1, num_criteria));
        exmatrix.slice_mut(s![1.., ..]).assign(matrix);
        println!("{}", exmatrix);
        for (i, criteria_type) in types.iter().enumerate() {
            if *criteria_type == CriteriaType::Profit {
                exmatrix[[0, i]] = *matrix.slice(s![.., i]).max()?;
            } else if *criteria_type == CriteriaType::Cost {
                exmatrix[[0, i]] = *matrix.slice(s![.., i]).min()?;
            }
        }
        println!("{}", exmatrix);

        let nmatrix = Sum::normalize(&exmatrix, types)?;
        let weighted_matrix = nmatrix.clone() * weights;

        let s = weighted_matrix.sum_axis(Axis(1));
        let k = s.slice(s![1..]).map(|x| x / s[0]);

        println!("\n\n{}\n\n{}\n\n{}", nmatrix, s, k);

        Ok(k)
    }
}

/// Ranks the alternatives using the COmbined Compromise SOlution (COCOSO) method.
///
/// The COCOSO method expects the decision matrix is normalized using the [`MinMax`](crate::normalization::MinMax)
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
/// where $k_{ia}$ represents the average of the sums of [`WeightedSum`] and [`WeightedProduct`]
/// scores, $k_{ib}$ represents the [`WeightedSum`] and [`WeightedProduct`] scores over the best
/// scores for each each method respectfully, and $k_{ic}$ represents the [`WeightedSum`],
/// [`WeightedProduct`] scores using the compromise strategy, and $n$ is the number of alternatives.
///
/// Lastly, we rank the alternatives as follows:
///
/// $$ k_i = (k_{ia}k_{ib}k_{ic})^{\frac{1}{3}} + \frac{1}{3}(k_{ia} + k_{ib} + k_{ic}) $$
///
///
/// # Arguments
///
/// * `matrix` - A normalized decision matrix of alternatives (alternatives x criteria).
/// * `weights` - A vector of weights for each criterion.
///
/// # Returns
///
/// * `Result<Array1<f64>, RankingError>` - A vector of scores for each alternative.
///
/// # Example
///
/// ```rust
/// use approx::assert_abs_diff_eq;
/// use mcdm::ranking::{Rank, Cocoso};
/// use mcdm::normalization::{MinMax, Normalize};
/// use ndarray::{array, Array2};
///
/// let matrix = array![
///     [2.9, 2.31, 0.56, 1.89],
///     [1.2, 1.34, 0.21, 2.48],
///     [0.3, 2.48, 1.75, 1.69]
/// ];
/// let weights = array![0.25, 0.25, 0.25, 0.25];
/// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
/// let normalized_matrix = MinMax::normalize(&matrix, &criteria_types).unwrap();
/// let ranking = Cocoso::rank(&normalized_matrix, &weights).unwrap();
/// assert_abs_diff_eq!(ranking, array![3.24754746, 1.14396494, 5.83576765], epsilon = 1e-5);
/// ```
pub struct Cocoso;

impl Rank for Cocoso {
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, RankingError> {
        if weights.len() != matrix.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        let l = 0.5;

        // Vectors of S and P
        let s = (matrix * weights).sum_axis(Axis(1));

        let mut p = Array1::zeros(matrix.nrows());
        //let mut p = matrix.mapv(|x| x.powf(*weights));
        for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
            let mut row_sum = 0.0;
            for (j, &value) in row.iter().enumerate() {
                row_sum += value.powf(weights[j]);
            }
            p[i] = row_sum;
        }
        // Calculate score strategies
        let ksi_a = (p.clone() + s.clone()) / (p.clone() + s.clone()).sum();
        let ksi_b = s.clone() / *s.min()? + p.clone() / *p.min()?;
        let ksi_c =
            (l * s.clone() + (1.0 - l) * p.clone()) / (l * s.max()? + (1.0 - l) * p.max()?);

        // Compute the performance score
        let ksi = (ksi_a.clone() * ksi_b.clone() * ksi_c.clone()).powf(1.0 / 3.0)
            + ((ksi_a.clone() + ksi_b.clone() + ksi_c.clone()) / 3.0f64);

        Ok(ksi)
    }
}

/// Ranks the alternatives using the COmbinative Distance-based ASessment (CODAS) method.
///
/// The CODAS method expects the decision matrix is normalized using the [`Linear`](crate::normalization::Linear)
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
/// # Example
///
/// ```rust
/// use approx::assert_abs_diff_eq;
/// use mcdm::ranking::{Rank, Codas};
/// use mcdm::normalization::{Linear, Normalize};
/// use ndarray::{array, Array2};
///
/// let matrix = array![
///     [2.9, 2.31, 0.56, 1.89],
///     [1.2, 1.34, 0.21, 2.48],
///     [0.3, 2.48, 1.75, 1.69]
/// ];
/// let weights = array![0.25, 0.25, 0.25, 0.25];
/// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
/// let normalized_matrix = Linear::normalize(&matrix, &criteria_types).unwrap();
/// let ranking = Codas::rank(&normalized_matrix, &weights).unwrap();
/// assert_abs_diff_eq!(ranking, array![-0.40977725, -1.15891275, 1.56869], epsilon = 1e-5);
/// ```
pub struct Codas;

impl Rank for Codas {
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, RankingError> {
        if weights.len() != matrix.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        let weighted_matrix = matrix * weights;
        // Compute the Negative Ideal Solution (NIS)
        let nis = weighted_matrix.fold_axis(ndarray::Axis(0), f64::INFINITY, |m, &v| m.min(v));

        let euclidean_distances = (&weighted_matrix - &nis)
            .mapv(|x| x.powi(2))
            .sum_axis(Axis(1))
            .mapv(|x| x.sqrt());
        let taxicab_distances = (&weighted_matrix - &nis)
            .mapv(|x| x.abs())
            .sum_axis(Axis(1));

        let mut assessment_matrix =
            Array2::zeros((weighted_matrix.nrows(), weighted_matrix.nrows()));

        for i in 0..weighted_matrix.nrows() {
            for j in 0..weighted_matrix.nrows() {
                let e = euclidean_distances[i] - euclidean_distances[j];
                let t = taxicab_distances[i] - taxicab_distances[j];
                assessment_matrix[[i, j]] = (e) + ((psi_with_default_tau(e)) * (t));
            }
        }

        Ok(assessment_matrix.sum_axis(Axis(1)))
    }
}

fn psi(x: f64, tau: f64) -> f64 {
    if x.abs() >= tau {
        1.0
    } else {
        0.0
    }
}

fn psi_with_default_tau(x: f64) -> f64 {
    psi(x, 0.02)
}

/// Ranks the alternatives using the COmplex PRoportional ASsessment (COPRAS) method.
///
/// The COPRAS method evaluates alternatives by separately considering the effects of maximizing
/// (beneficial) and minimizing (non-beneficial) index values of attributes. This approach allows
/// COPRAS to assess the impact of each type of crition independently, ensuring both positive
/// contributions and cost factors are accounted for in the final ranking. This separation provides
/// a more balanced and accurate assessment of each alternative.
///
/// Start by calculating the normalized decision matrix, $r_{ij}$, using the [`Sum`] method, but treat each
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
/// # Example
///
/// ```rust
/// use approx::assert_abs_diff_eq;
/// use mcdm::ranking::{RankWithCriteriaType, Copras};
/// use mcdm::normalization::{Linear, Normalize};
/// use ndarray::{array, Array2};
///
/// let matrix = array![
///     [2.9, 2.31, 0.56, 1.89],
///     [1.2, 1.34, 0.21, 2.48],
///     [0.3, 2.48, 1.75, 1.69]
/// ];
/// let weights = array![0.25, 0.25, 0.25, 0.25];
/// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
/// let ranking = Copras::rank(&matrix, &criteria_types, &weights).unwrap();
/// assert_abs_diff_eq!(ranking, array![1.0, 0.6266752, 0.92104753], epsilon = 1e-5);
/// ```
pub struct Copras;

impl RankWithCriteriaType for Copras {
    fn rank(
        matrix: &Array2<f64>,
        types: &[CriteriaType],
        weights: &Array1<f64>,
    ) -> Result<Array1<f64>, RankingError> {
        let (num_alternatives, num_criteria) = matrix.dim();

        if num_alternatives == 0 || num_criteria == 0 {
            return Err(RankingError::EmptyMatrix);
        }

        if types.len() != num_criteria {
            return Err(RankingError::DimensionMismatch);
        }

        let normalized_matrix = Sum::normalize(matrix, &CriteriaType::profits(types.len()))?;
        let weighted_matrix = normalized_matrix * weights;

        let (sum_normalized_profit, sum_normalized_cost): (Array1<f64>, Array1<f64>) =
            types.iter().zip(weighted_matrix.axis_iter(Axis(1))).fold(
                (
                    Array1::zeros(num_alternatives),
                    Array1::zeros(num_alternatives),
                ),
                |(profit, cost), (criteria_type, row)| match criteria_type {
                    CriteriaType::Profit => (profit + row, cost),
                    CriteriaType::Cost => (profit, cost + row),
                },
            );

        let min_sm = *sum_normalized_cost.clone().min()?;
        let q = sum_normalized_profit
            + ((min_sm * sum_normalized_cost.clone())
                / (sum_normalized_cost.clone() * sum_normalized_cost.mapv(|x| min_sm / x)));

        let max_q = *q.clone().max()?;
        let q = q / max_q;

        Ok(q)
    }
}

/// Ranks the alternatives using the Multi-Attributive Border Approximation Area Comparison (MABAC)
/// method.
///
/// The MABAC method expects the decision matrix is normalized using the [`MinMax`](crate::normalization::MinMax)
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
/// * `matrix` - A normalized decision matrix of alternatives (alternatives x criteria).
/// * `weights` - A vector of weights for each criterion.
///
/// # Returns
///
/// * `Result<Array1<f64>, RankingError>` - A vector of scores for each alternative.
///
/// # Example
///
/// ```rust
/// use approx::assert_abs_diff_eq;
/// use mcdm::ranking::{Rank, Mabac};
/// use mcdm::normalization::{MinMax, Normalize};
/// use ndarray::{array, Array2};
///
/// let matrix = array![
///     [2.9, 2.31, 0.56, 1.89],
///     [1.2, 1.34, 0.21, 2.48],
///     [0.3, 2.48, 1.75, 1.69]
/// ];
/// let weights = array![0.25, 0.25, 0.25, 0.25];
/// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
/// let normalized_matrix = MinMax::normalize(&matrix, &criteria_types).unwrap();
/// let ranking = Mabac::rank(&normalized_matrix, &weights).unwrap();
/// assert_abs_diff_eq!(ranking, array![-0.01955314, -0.31233795,  0.52420052], epsilon = 1e-5);
/// ```
pub struct Mabac;

impl Rank for Mabac {
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, RankingError> {
        if weights.len() != matrix.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        // Calculation of the elements from the weighted matrix
        let weighted_matrix = (matrix + 1.0) * weights;

        // Determining the border approximation area matrix
        let g = weighted_matrix
            .map_axis(Axis(0), |row| row.product())
            .mapv(|x| x.powf(1.0 / matrix.nrows() as f64));

        // Calculation of the distance border approximation area
        let q = weighted_matrix - g;

        Ok(q.sum_axis(Axis(1)))
    }
}

/// Ranks the alternatives using the TOPSIS method.
///
/// The TOPSIS method expects the decision matrix is normalized using the [`MinMax`](crate::normalization::MinMax)
/// method. Then computes a weighted matrix $v_{ij}$ using
///
/// $$ v_{ij} = x_{ij}{w_j} $$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
/// (column), and $w_j$ is the weight of the $j$th criterion.
///
/// We then derive a positive ideal solution (PIS) and a negative ideal solution (NIS). The PIS is
/// calculated as the maximum value for each criterion, and the NIS is the minimum value for each
/// criterion.
///
/// $$ v_j^+ = \left\\{v_1^+, v_2^+, \dots, v_n^+\right\\} = \max_j v_{ij} $$
/// $$ v_j^- = \left\\{v_1^-, v_2^-, \dots, v_n^-\right\\} = \min_j v_{ij} $$
///
/// where $v_j^+$ is the positive ideal solution, $v_j^-$ is the negative ideal solution, $n$ is the
/// number of criteria, and $i$ is the alternative index
///
/// Finally we determine the distance to the PIS ($D_i^+$) and NIS ($D_i^-$). The distance to the
/// PIS is calculated as the square root of the sum of the squares of the differences between the
/// weighted matrix row and the PIS. The distance to the NIS is calculated as the square root of the
/// sum of the squares of the differences between the weighted matrix row and the NIS as follows:
///
/// $$ D_i^+ = \sqrt{ \sum_{j=1}^{n} (v_{ij} - v_j^+)^2 } $$
/// $$ D_i^- = \sqrt{ \sum_{j=1}^{n} (v_{ij} - v_j^-)^2 } $$
///
/// # Arguments
///
/// * `matrix` - A normalized decision matrix of alternatives (alternatives x criteria).
/// * `weights` - A vector of weights for each criterion.
///
/// # Returns
///
/// * `Result<Array1<f64>, RankingError>` - A vector of scores for each alternative.
///
/// # Example
///
/// ```rust
/// use approx::assert_abs_diff_eq;
/// use mcdm::ranking::{Rank, TOPSIS};
/// use mcdm::normalization::{MinMax, Normalize};
/// use ndarray::{array, Array2};
///
/// let matrix = array![
///     [2.9, 2.31, 0.56, 1.89],
///     [1.2, 1.34, 0.21, 2.48],
///     [0.3, 2.48, 1.75, 1.69]
/// ];
/// let weights = array![0.25, 0.25, 0.25, 0.25];
/// let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
/// let normalized_matrix = MinMax::normalize(&matrix, &criteria_types).unwrap();
/// let ranking = TOPSIS::rank(&normalized_matrix, &weights).unwrap();
/// assert_abs_diff_eq!(ranking, array![0.47089549, 0.27016783, 1.0], epsilon = 1e-5);
/// ```
pub struct TOPSIS;

impl Rank for TOPSIS {
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, RankingError> {
        if weights.len() != matrix.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        if weights.iter().any(|x| *x == 0.0) {
            return Err(RankingError::InvalidValue);
        }

        let num_rows = matrix.nrows();

        let weighted_matrix = matrix * weights;

        // Compute the Positive Ideal Solution (PIS) and Negative Ideal Solution (NIS)
        let pis = weighted_matrix.fold_axis(ndarray::Axis(0), f64::NEG_INFINITY, |m, &v| m.max(v));
        let nis = weighted_matrix.fold_axis(ndarray::Axis(0), f64::INFINITY, |m, &v| m.min(v));

        // Calculate the distance to PIS (Dp) and NIS (Dm)
        let mut distance_to_pis = Array1::zeros(num_rows);
        let mut distance_to_nis = Array1::zeros(num_rows);

        for i in 0..num_rows {
            let row = weighted_matrix.row(i);

            let dp = (&row - &pis).mapv(|x| x.powi(2)).sum().sqrt();
            distance_to_pis[i] = dp;

            let dn = (&row - &nis).mapv(|x| x.powi(2)).sum().sqrt();
            distance_to_nis[i] = dn;
        }

        // Calculate the relative closeness to the ideal solution
        Ok(&distance_to_nis / (&distance_to_nis + &distance_to_pis))
    }
}

/// Computes the Weighted Product Model (WPM) preference values for alternatives.
///
/// The WPM model expects the decision matrix is normalized using the [`Sum`] method. Then computes
/// the ranking using:
///
/// $$ WPM = \prod_{j=1}^n(x_{ij})^{w_j} $$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
/// (column) with $n$ total criteria, and $w_j$ is the weight of the $j$th criterion.
///
/// # Arguments
///
/// * `matrix` - A normalized decision matrix where each row represents an alternative and each
///   column represents a criterion.
/// * `weights` - A 1D array of weights corresponding to the relative importance of each criterion.
///
/// # Returns
///
/// * `Array1<f64>` - A 1D array containing the preference values for each alternative.
///
/// # Example
///
/// ```rust
/// use approx::assert_abs_diff_eq;
/// use mcdm::ranking::{Rank, WeightedProduct};
/// use mcdm::normalization::{Normalize, Sum};
/// use mcdm::CriteriaType;
/// use ndarray::{array};
///
/// let matrix = array![
///     [2.9, 2.31, 0.56, 1.89],
///     [1.2, 1.34, 0.21, 2.48],
///     [0.3, 2.48, 1.75, 1.69]
/// ];
/// let weights = array![0.25, 0.25, 0.25, 0.25];
/// let criteria_type = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
/// let normalized_matrix = Sum::normalize(&matrix, &criteria_type).unwrap();
/// let ranking = WeightedProduct::rank(&normalized_matrix, &weights).unwrap();
/// assert_abs_diff_eq!(
///     ranking,
///     array![0.21711531, 0.17273414, 0.53281425],
///     epsilon = 1e-5
/// );
/// ```
pub struct WeightedProduct;

impl Rank for WeightedProduct {
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, RankingError> {
        if weights.len() != matrix.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        // Compute the weighted matrix by raising each element of the decision matrix to the power
        // of the corresponding weight.
        let mut weighted_matrix = Array2::zeros(matrix.dim());

        // NOTE: I'm sure there is an idiomatic way to do this, but I can't seem to figure it out.
        for (i, row) in matrix.axis_iter(Axis(0)).enumerate() {
            for (j, &value) in row.iter().enumerate() {
                weighted_matrix[[i, j]] = value.powf(weights[j]);
            }
        }

        // Compute the product of each row
        Ok(weighted_matrix.map_axis(Axis(1), |row| row.product()))
    }
}

/// Rank the alternatives using the Weighted Sum Model.
///
/// The `WeightedSum` method ranks alternatives based on the weighted sum of their criteria values.
/// Each alternative's score is calculated by multiplying its criteria values by the corresponding
/// weights and summing the results. The decision matrix is expected to be normalized using the
/// [`Sum`] method.
///
/// $$ WSM = \sum_{j=1}^n x_{ij}{w_j} $$
///
/// where $x_{ij}$ is the $i$th element of the alternative (row), $j$th elements of the criterion
/// (column) with $n$ total criteria, and $w_j$ is the weight of the $j$th criterion.
///
///
/// # Arguments
///
/// * `matrix` - A normalized decision matrix where each row represents an alternative and each
///   column represents a criterion.
/// * `weights` - A 1D array of weights corresponding to the relative importance of each criterion.
///
/// # Returns
///
/// * `Result<Array1<f64>, RankingError>` - A 1D array containing the preference values for each
///   alternative, or an error if the ranking process fails.
///
/// # Example
///
/// ```rust
/// use approx::assert_abs_diff_eq;
/// use mcdm::ranking::{WeightedSum, Rank};
/// use ndarray::{array, Array1, Array2};
///
/// let matrix = array![[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]];
/// let weights = array![0.6, 0.4];
/// let ranking = WeightedSum::rank(&matrix, &weights).unwrap();
/// assert_abs_diff_eq!(ranking, array![0.44, 0.5, 0.58], epsilon = 1e-5);
/// ```
pub struct WeightedSum;

impl Rank for WeightedSum {
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, RankingError> {
        if weights.len() != matrix.ncols() {
            return Err(RankingError::DimensionMismatch);
        }

        let weighted_matrix = matrix * weights;
        Ok(weighted_matrix.sum_axis(Axis(1)))
    }
}
