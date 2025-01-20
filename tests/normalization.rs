use approx::assert_relative_eq;
use mcdm::{errors::McdmError, normalization::Normalize, CriteriaTypes};
use nalgebra::{dmatrix, DMatrix};

mod enhanced_accuracy_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_enhanced_accuracy(&criteria_types)?;
        let expected_matrix = dmatrix![
            0.25714286, 0.87022901, 0.56410256, 0.7979798;
            0.74285714, 0.12977099, 0.43589744, 0.2020202;
            1.0, 1.0, 1.0, 1.0
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod linear_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_linear(&criteria_types)?;
        let expected_matrix = dmatrix![
            0.10344828, 0.93145161, 0.32, 0.89417989;
            0.25, 0.54032258, 0.12, 0.68145161;
            1.0, 1.0, 1.0, 1.0
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod logarithmic_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_logarithmic(&criteria_types)?;
        let expected_matrix = dmatrix![
            -11.86325315, 0.4107828, 0.36677631, 0.34620508;
            -1.61708916, 0.14359391, 0.98722036, 0.28056765;
            14.48034231, 0.44562329, -0.35399666, 0.37322727
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod nonlinear_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_nonlinear(&criteria_types)?;
        let expected_matrix = dmatrix![
            0.00110706, 0.86760211, 0.1024, 0.7149484;
            0.015625, 0.29194849, 0.0144, 0.31644998;
            1.0, 1.0, 1.0, 1.0
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod max_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_max(&criteria_types)?;
        let expected_matrix = dmatrix![
            0.0, 0.93145161, 0.32, 0.23790323;
            0.5862069, 0.54032258, 0.12, 0.0;
            0.89655172, 1.0, 1.0, 0.31854839
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod minmax_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_min_max(&criteria_types)?;
        let expected_matrix = dmatrix![
            0.0, 0.85087719, 0.22727273, 0.74683544;
            0.65384615, 0.0, 0.0, 0.0;
            1.0, 1.0, 1.0, 1.0
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod ocra_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_ocra(&criteria_types)?;
        let expected_matrix = dmatrix![
            0.0, 0.7238806, 1.66666667, 0.34911243;
            5.66666667, 0.0, 0.0, 0.0;
            8.66666667, 0.85074627, 7.33333333, 0.46745562
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod rim_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_range = dmatrix![0.1, 3.0; 1.0, 2.48; 0.0, 1.9; 0.69, 3.1];
        let reference_ideal = dmatrix![0.1, 0.1; 2.48, 2.48; 1.9, 1.9; 0.69, 0.69];
        let normalized_matrix = matrix.normalize_rim(&criteria_range, &reference_ideal)?;
        let expected_matrix = dmatrix![
            0.03448276, 0.88513514, 0.29473684, 0.50207469;
            0.62068966, 0.22972973, 0.11052632, 0.25726141;
            0.93103448, 1.0,        0.92105263, 0.58506224
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod spotis_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let bounds = dmatrix![0.1, 3.0; 1.0, 2.48; 0.0, 1.9; 0.69, 3.1];
        let normalized_matrix = matrix.normalize_spotis(&criteria_types, &bounds)?;
        let expected_matrix = dmatrix![
            0.96551724, 0.11486486, 0.70526316, 0.49792531;
            0.37931034, 0.77027027, 0.88947368, 0.74273859;
            0.06896552, 0.0       , 0.07894737, 0.41493776
        ];

        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod sum_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix: DMatrix<f64> = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_sum(&criteria_types)?;
        let expected_matrix = dmatrix![
            0.07643312, 0.37683524, 0.22222222, 0.34716919;
            0.18471338, 0.21859706, 0.08333333, 0.26457652;
            0.7388535, 0.4045677, 0.69444444, 0.3882543
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod vector_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_vector(&criteria_types)?;
        let expected_matrix = dmatrix![
            0.08017585, 0.63383849, 0.30280447, 0.46710009;
            0.61938311, 0.3676812, 0.11355167, 0.30074509;
            0.90484578, 0.68048461, 0.94626396, 0.52349161
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}

mod zavadskas_turskis_tests {
    use super::*;

    #[test]
    fn test_normalize() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_zavadskas_turskis(&criteria_types)?;
        let expected_matrix = dmatrix![
            -7.66666667, 0.93145161, 0.32, 0.8816568;
            -2.0, 0.54032258, 0.12, 0.53254438;
            1.0, 1.0, 1.0, 1.0
        ];
        assert_relative_eq!(normalized_matrix, expected_matrix, epsilon = 1e-5);

        Ok(())
    }
}
