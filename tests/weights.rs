use approx::assert_abs_diff_eq;
use mcdm::normalization::*;
use mcdm::weights::*;
use mcdm::CriteriaType;
use ndarray::array;

mod angular_tests {
    use super::*;

    #[test]
    fn test_weight() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let weights = Angular::weight(&matrix).unwrap();
        let expected_weights = array![0.37183274, 0.14136016, 0.39029729, 0.09650981];
        assert_abs_diff_eq!(weights, expected_weights, epsilon = 1e-5);
    }
}

mod critic_tests {
    use super::*;

    #[test]
    fn test_weight() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::profits(matrix.ncols());
        let normalized_matrix = MinMax::normalize(&matrix, &criteria_types).unwrap();
        let weights = Critic::weight(&normalized_matrix).unwrap();
        let expected_weights = array![0.22514451, 0.21769531, 0.24375671, 0.31340347];
        assert_abs_diff_eq!(weights, expected_weights, epsilon = 1e-5);
    }
}

mod entropy_tests {
    use super::*;

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

mod merec_tests {
    use super::*;
    use mcdm::normalization::{Linear, Normalize};
    use mcdm::CriteriaType;

    #[test]
    fn test_weight() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];

        // NOTE: This is the "oposite" of the criteria types we'd expect to use for other methods.
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix =
            Linear::normalize(&matrix, &CriteriaType::switch(criteria_types)).unwrap();
        let weights = Merec::weight(&normalized_matrix).unwrap();
        let expected_weights = array![0.40559526, 0.14185966, 0.37609753, 0.07644754];
        assert_abs_diff_eq!(weights, expected_weights, epsilon = 1e-5);
    }
}

mod gini_tests {
    use super::*;

    #[test]
    fn test_weight() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let weights = Gini::weight(&matrix).unwrap();
        let expected_weights = array![0.38917745, 0.12248175, 0.40248266, 0.08585814];
        assert_abs_diff_eq!(weights, expected_weights, epsilon = 1e-5);
    }
}

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

mod variance_tests {
    use super::*;

    #[test]
    fn test_weight() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::profits(matrix.ncols());
        let normalized_matrix = MinMax::normalize(&matrix, &criteria_types).unwrap();
        let weights = Variance::weight(&normalized_matrix).unwrap();
        let expected_weights = array![0.23572429, 0.26602392, 0.25117527, 0.24707653];
        assert_abs_diff_eq!(weights, expected_weights, epsilon = 1e-5);
    }
}
