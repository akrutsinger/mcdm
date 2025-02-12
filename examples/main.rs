use mcdm::{CriteriaTypes, McdmError, Normalize, Rank, Weight};
use nalgebra::dmatrix;

fn main() -> Result<(), McdmError> {
    // Define the decision matrix (alternatives x criteria)
    let alternatives = dmatrix![4.0, 7.0, 8.0; 2.0, 9.0, 6.0; 3.0, 6.0, 9.0];
    //let criteria_types = CriteriaType::from(vec![-1, 1, 1])?;
    let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1])?;

    // Apply normalization using Min-Max
    let normalized_matrix = alternatives.normalize_min_max(&criteria_types)?;

    // Alternatively, use equal weights
    let equal_weights = normalized_matrix.weight_equal()?;

    // Apply the TOPSIS method for ranking
    let ranking = normalized_matrix.rank_topsis(&equal_weights)?;

    // Output the ranking
    println!("Ranking: {:.3}", ranking);

    Ok(())
}
