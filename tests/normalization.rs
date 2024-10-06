use approx::assert_abs_diff_eq;
use mcdm::normalization::*;
use mcdm::CriteriaType;
use ndarray::array;

mod enhanced_accuracy_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = EnhancedAccuracy::normalize(&matrix, &criteria_types).unwrap();
        let expected_matrix = array![
            [0.25714286, 0.87022901, 0.56410256, 0.7979798],
            [0.74285714, 0.12977099, 0.43589744, 0.2020202],
            [1.0, 1.0, 1.0, 1.0]
        ];
        assert_abs_diff_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
    }
}

mod linear_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = Linear::normalize(&matrix, &criteria_types).unwrap();
        let expected_matrix = array![
            [0.10344828, 0.93145161, 0.32, 0.89417989],
            [0.25, 0.54032258, 0.12, 0.68145161],
            [1.0, 1.0, 1.0, 1.0]
        ];
        assert_abs_diff_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
    }
}

mod logarithmic_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = Logarithmic::normalize(&matrix, &criteria_types).unwrap();
        let expected_matrix = array![
            [-11.86325315, 0.4107828, 0.36677631, 0.34620508],
            [-1.61708916, 0.14359391, 0.98722036, 0.28056765],
            [14.48034231, 0.44562329, -0.35399666, 0.37322727]
        ];
        assert_abs_diff_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
    }
}

mod nonlinear_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = NonLinear::normalize(&matrix, &criteria_types).unwrap();
        let expected_matrix = array![
            [0.00110706, 0.86760211, 0.1024, 0.7149484],
            [0.015625, 0.29194849, 0.0144, 0.31644998],
            [1.0, 1.0, 1.0, 1.0]
        ];
        assert_abs_diff_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
    }
}

mod max_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = Max::normalize(&matrix, &criteria_types).unwrap();
        let expected_matrix = array![
            [0.0, 0.93145161, 0.32, 0.23790323],
            [0.5862069, 0.54032258, 0.12, 0.0],
            [0.89655172, 1.0, 1.0, 0.31854839]
        ];
        assert_abs_diff_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
    }
}

mod minmax_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = MinMax::normalize(&matrix, &criteria_types).unwrap();
        let expected_matrix = array![
            [0.0, 0.85087719, 0.22727273, 0.74683544],
            [0.65384615, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0]
        ];
        assert_abs_diff_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
    }
}

mod sum_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = Sum::normalize(&matrix, &criteria_types).unwrap();
        let expected_matrix = array![
            [0.07643312, 0.37683524, 0.22222222, 0.34716919],
            [0.18471338, 0.21859706, 0.08333333, 0.26457652],
            [0.7388535, 0.4045677, 0.69444444, 0.3882543]
        ];
        assert_abs_diff_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
    }
}

mod vector_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = Vector::normalize(&matrix, &criteria_types).unwrap();
        let expected_matrix = array![
            [0.08017585, 0.63383849, 0.30280447, 0.46710009],
            [0.61938311, 0.3676812, 0.11355167, 0.30074509],
            [0.90484578, 0.68048461, 0.94626396, 0.52349161]
        ];
        assert_abs_diff_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
    }
}

mod zavadskas_turskis_tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let matrix = array![
            [2.9, 2.31, 0.56, 1.89],
            [1.2, 1.34, 0.21, 2.48],
            [0.3, 2.48, 1.75, 1.69]
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1]).unwrap();
        let normalized_matrix = ZavadskasTurskis::normalize(&matrix, &criteria_types).unwrap();
        let expected_matrix = array![
            [-7.66666667, 0.93145161, 0.32, 0.8816568],
            [-2.0, 0.54032258, 0.12, 0.53254438],
            [1.0, 1.0, 1.0, 1.0]
        ];
        assert_abs_diff_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);
    }
}
