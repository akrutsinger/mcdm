use approx::assert_abs_diff_eq;
use mcdm::methods::*;
use ndarray::array;

mod topsis_tests {
    use super::*;
    use mcdm::normalization::{MinMax, Normalize};

    #[test]
    fn test_rank() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let weights = array![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = MinMax::normalize(&matrix, &criteria_types).unwrap();
        let ranking = TOPSIS::rank(&normalized_matrix, &weights).unwrap();
        assert_abs_diff_eq!(ranking, array![0.47089549, 0.27016783, 1.0], epsilon = 1e-5);
    }

    #[test]
    fn test_with_zero_weights() {
        let matrix = array![[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]];
        let weights = array![0.0, 0.0];
        let ranking = TOPSIS::rank(&matrix, &weights);
        assert!(ranking.is_err());
    }

    #[test]
    fn test_rank_with_invalid_dimensions() {
        let matrix = array![[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]];
        let weights = array![0.6];
        let ranking = TOPSIS::rank(&matrix, &weights);
        assert!(ranking.is_err());
    }
}

mod weighted_product_tests {
    use super::*;
    use mcdm::normalization::{Normalize, Sum};
    use mcdm::CriteriaType;

    #[test]
    fn test_rank() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let weights = array![0.25, 0.25, 0.25, 0.25];
        let criteria_type = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = Sum::normalize(&matrix, &criteria_type).unwrap();
        let ranking = WeightedProduct::rank(&normalized_matrix, &weights).unwrap();
        assert_abs_diff_eq!(
            ranking,
            array![0.21711531, 0.17273414, 0.53281425],
            epsilon = 1e-5
        );
    }

    #[test]
    fn test_rank_with_invalid_dimensions() {
        let matrix = array![[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]];
        let weights = array![0.6];
        let ranking = WeightedSum::rank(&matrix, &weights);
        assert!(ranking.is_err());
    }
}

mod weighted_sum_tests {
    use super::*;
    #[test]
    fn test_rank() {
        let matrix = array![[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]];
        let weights = array![0.6, 0.4];
        let ranking = WeightedSum::rank(&matrix, &weights).unwrap();
        assert_abs_diff_eq!(ranking, array![0.44, 0.5, 0.58], epsilon = 1e-5);
    }

    #[test]
    fn test_with_zero_weights() {
        let matrix = array![[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]];
        let weights = array![0.0, 0.0];
        let ranking = WeightedSum::rank(&matrix, &weights).unwrap();
        assert_eq!(ranking, array![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_rank_with_single_criterion() {
        let matrix = array![[0.2], [0.5], [0.9]];
        let weights = array![1.0];
        let ranking = WeightedSum::rank(&matrix, &weights).unwrap();
        assert_eq!(ranking, array![0.2, 0.5, 0.9]);
    }

    #[test]
    fn test_rank_with_invalid_dimensions() {
        let matrix = array![[0.2, 0.8], [0.5, 0.5], [0.9, 0.1]];
        let weights = array![0.6];
        let ranking = WeightedSum::rank(&matrix, &weights);
        assert!(ranking.is_err());
    }
}
