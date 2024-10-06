use approx::assert_abs_diff_eq;
use mcdm::weights::*;
use ndarray::array;

mod equal_tests {
    use super::*;

    #[test]
    fn test_weight() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let weights = Equal::weight(&matrix).unwrap();
        let expected_weights = array![0.25, 0.25, 0.25, 0.25];
        assert_abs_diff_eq!(weights, expected_weights, epsilon = 1e-5);
    }
}

mod entropy_tests {
    use super::*;
    use mcdm::normalization::{Normalize, Sum};
    use mcdm::CriteriaType;

    #[test]
    fn test_weight() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::profits(matrix.ncols());
        let normalized_matrix = Sum::normalize(&matrix, &criteria_types).unwrap();
        let weights = Entropy::weight(&normalized_matrix).unwrap();
        let expected_weights = array![0.45009235, 0.05084365, 0.4778931, 0.0211709];
        assert_abs_diff_eq!(weights, expected_weights, epsilon = 1e-5);
    }
}

mod merec_tests {}

mod standard_deviation_tests {
    use super::*;

    #[test]
    fn test_weight() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let weights = StandardDeviation::weight(&matrix).unwrap();
        let expected_weights = array![0.41871179, 0.19503155, 0.25600523, 0.13025143];
        assert_abs_diff_eq!(weights, expected_weights, epsilon = 1e-5);
    }
}
