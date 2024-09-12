use crate::errors::MethodError;
use ndarray::{Array1, Array2};

/// A trait for ranking alternatives in Multiple-Criteria Decision Making (MCDM).
///
/// The [Rank] trait defines a method used to rank alternatives based on a normalized decision
/// matrix and a set of weights for the criteria. The ranking process evaluates how well each
/// alternative performs across the criteria, considering the relative importance of each criterion
/// as given by the `weights` array.
///
/// Higher preference values indicate better alternatives. The specific ranking method used (such as
/// [TOPSIS] or others) will depend on the implementation of this trait.
///
/// # Example
///
/// Here’s an example of ranking alternatives using the [Rank] trait:
///
/// ```rust
/// use mcdm::{methods::TOPSIS, Rank};
/// use ndarray::{array, Array1, Array2};
///
/// let normalized_matrix: Array2<f64> = array![[0.8, 0.6], [0.5, 0.9], [0.3, 0.7]];
/// let weights: Array1<f64> = array![0.6, 0.4];
/// let rankings = TOPSIS::rank(&normalized_matrix, &weights).unwrap();
/// println!("Rankings: {:?}", rankings);
/// ```
///
/// # Arguments
///
/// * `matrix` - A normalized decision matrix (`Array2<f64>`), where rows represent alternatives
/// and columns represent criteria. This matrix must already be normalized, as the ranking process
/// assumes that the values are comparable across criteria.
/// * `weights` - A 1D array (`Array1<f64>`) representing the relative importance of each criterion.
/// The length of this array must match the number of columns in `matrix`.
///
/// # Returns
///
/// This method returns a `Result<Array1<f64>, MethodError>`, where:
///
/// - `Array1<f64>` is a 1D array containing the preference values for each alternative. Higher
/// values indicate better alternatives.
/// - [MethodError] is returned if the ranking process fails (e.g., due to a mismatch between the
/// dimensions of `matrix` and `weights`, or other calculation errors).
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
    /// column represents a criterion.
    /// * `weights` - A 1D array of weights corresponding to the relative importance of each criterion.
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, MethodError>` - A 1D array of preference values, or an error if the
    /// ranking process fails.
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, MethodError>;
}

/// Ranks the alternatives using the TOPSIS method.
///
/// # Arguments
///
/// * `matrix` - A normalized decision matrix of alternatives (alternatives x criteria).
/// * `weights` - A vector of weights for each criterion.
///
/// # Returns
///
/// * `Result<Array1<f64>, MethodError>` - A vector of scores for each alternative.
///
/// # Example
///
/// ```rust
/// use mcdm::{errors, methods::topsis::TOPSIS};
/// use ndarray::{array, Array2};
///
/// fn main() -> Result<(), errors::MethodError> {
///     // Define the decision matrix (alternatives x criteria)
///     let matrix = array![
///         [0.1, 0.1, 0.1, 0.1],
///         [0.2, 0.2, 0.2, 0.2],
///         [0.3, 0.3, 0.3, 0.3]
///     ];
///
///     // Define the weights
///     let weights = array![0.25, 0.25, 0.25, 0.25];
///
///     // Rank the alternatives
///     let ranking = TOPSIS::rank(&matrix, &weights)?;
///
///     // Output the ranking
///     println!("Ranking: {:?}", ranking);
///		// Output: Ranking: [0.5, 0.5, 0.5]
///     Ok(())
/// }
/// ```
pub struct TOPSIS;

impl Rank for TOPSIS {
    fn rank(matrix: &Array2<f64>, weights: &Array1<f64>) -> Result<Array1<f64>, MethodError> {
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
