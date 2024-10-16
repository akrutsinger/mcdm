use crate::errors::RankingError;
use ndarray::{Array1, Array2, Axis};
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
///
/// # Arguments
///
/// * `matrix` - A normalized decision matrix (`Array2<f64>`), where rows represent alternatives
///   and columns represent criteria. This matrix must already be normalized, as the ranking process
///   assumes that the values are comparable across criteria.
/// * `weights` - A 1D array (`Array1<f64>`) representing the relative importance of each criterion.
///   The length of this array must match the number of columns in `matrix`.
///
/// # Returns
///
/// This method returns a `Result<Array1<f64>, RankingError>`, where:
///
/// - `Array1<f64>` is a 1D array containing the preference values for each alternative. Higher
///   values indicate better alternatives.
/// - [`RankingError`] is returned if the ranking process fails (e.g., due to a mismatch between the
///   dimensions of `matrix` and `weights`, or other calculation errors).
pub trait Rank {
    /// Ranks the alternatives in a normalized decision matrix based on the provided criteria
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
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, RankingError>` - A 1D array of preference values, or an error if the
    ///   ranking process fails.
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, RankingError>;
}

/// Ranks the alternatives using the COmbined Compromise SOlution (COCOSO) method.
///
/// The COCOSO method expects the decision matrix is normalized using the [`MinMax`](crate::normalization::MinMax)
/// method. Then calculate the weighted sum of the comparision sequence and the total power weight
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
/// The WPM model expects the decision matrix is normalized using the [`Sum`](crate::normalization::Sum)
/// method. Then computes the ranking using:
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
/// [`Sum`](crate::normalization::Sum) method.
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
