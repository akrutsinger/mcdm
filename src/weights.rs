use crate::errors::WeightingError;
use ndarray::{Array1, Array2};

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
/// use mcdm::{weights::Equal, Weight};
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
/// alternatives and columns represent criteria.
///
/// # Returns
///
/// This method returns a `Result<Array1<f64>, WeightingError>`, where:
///
/// * `Array1<f64>` is a 1D array of weights corresponding to each criterion.
/// * `WeightingError` is returned if an error occurs while calculating the weights (e.g., an
/// invalid matrix shape or calculation failure).
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
    /// columns represent criteria.
    ///
    /// # Returns
    ///
    /// * `Result<Array1<f64>, WeightingError>` - A vector of weights for each criterion, or error
    /// if the weighting calculation fails.
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
