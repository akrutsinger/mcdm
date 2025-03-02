use mcdm::{CriteriaTypes, McdmError, Normalize, Rank, Weight};
use nalgebra::dmatrix;

fn main() -> Result<(), McdmError> {
    // Define the decision matrix (alternatives x criteria)
    let alternatives = dmatrix![
        4.0, 7.0, 8.0;  // Alternative A
        2.0, 9.0, 6.0;  // Alternative B
        3.0, 6.0, 9.0   // Alternative C
    ];

    // Define the criteria types: 1 for benefit (maximize), -1 for cost (minimize)
    let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1])?;

    // Step 1: Normalize the decision matrix
    let normalized_matrix = alternatives.normalize_min_max(&criteria_types)?;

    // Step 2: Apply equal weights to all criteria.
    let equal_weights = normalized_matrix.weight_equal()?;

    // Step 3: Rank alternatives using TOPSIS
    let ranking = normalized_matrix.rank_topsis(&equal_weights)?;

    // Output the ranking (higher values indicate more preferred alternative)
    println!("Ranking: {:.3}", ranking);

    Ok(())
}
