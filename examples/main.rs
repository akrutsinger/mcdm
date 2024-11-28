use mcdm::{
    errors::McdmError,
    normalization::{MinMax, Normalize},
    ranking::{Rank, Topsis},
    weighting::{Equal, Weight},
    CriteriaType,
};
use nalgebra::dmatrix;

fn main() -> Result<(), McdmError> {
    // Define the decision matrix (alternatives x criteria)
    let alternatives = dmatrix![4.0, 7.0, 8.0; 2.0, 9.0, 6.0; 3.0, 6.0, 9.0];
    let criteria_types = CriteriaType::from(vec![-1, 1, 1])?;

    // Apply normalization using Min-Max
    let normalized_matrix = MinMax::normalize(&alternatives, &criteria_types)?;

    // Alternatively, use equal weights
    let equal_weights = Equal::weight(&normalized_matrix)?;

    // Apply the TOPSIS method for ranking
    let ranking = Topsis::rank(&normalized_matrix, &equal_weights)?;

    // Output the ranking
    println!("Ranking: {:.3?}", ranking);

    Ok(())
}
