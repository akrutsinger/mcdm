use mcdm::{
    errors::McdmError,
    methods::{Rank, TOPSIS},
    normalization::{MinMax, Normalize},
    weights::{Equal, Weight},
};
use ndarray::{array, Array1, Array2};

fn main() -> Result<(), McdmError> {
    // Define the decision matrix (alternatives x criteria)
    let alternatives: Array2<f64> = array![[4.0, 7.0, 8.0], [2.0, 9.0, 6.0], [3.0, 6.0, 9.0]];
    let criteria_types: Array1<i8> = array![-1, 1, 1];

    // Apply normalization using Min-Max
    let normalized_matrix = MinMax::normalize(&alternatives, &criteria_types)?;

    // Alternatively, use equal weights
    let equal_weights = Equal::weight(&normalized_matrix)?;

    // Apply the TOPSIS method for ranking
    let ranking = TOPSIS::rank(&normalized_matrix, &equal_weights)?;

    // Output the ranking
    println!("Ranking: {:.3?}", ranking);

    Ok(())
}
