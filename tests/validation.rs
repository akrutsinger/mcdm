mod bounds_validation_tests {
    use mcdm::errors::ValidationError;
    use mcdm::validation::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_is_bounds_valid() {
        let bounds = DMatrix::from_row_slice(3, 2, &[0.0, 10.0, 1.0, 9.0, 2.0, 8.0]);
        assert!(bounds.is_bounds_valid().is_ok());
    }

    #[test]
    fn test_is_bounds_invalid_shape() {
        let bounds = DMatrix::from_row_slice(3, 3, &[0.0, 10.0, 5.0, 1.0, 9.0, 5.0, 2.0, 8.0, 5.0]);
        assert!(matches!(
            bounds.is_bounds_valid().unwrap_err(),
            ValidationError::InvalidShape
        ));
    }

    #[test]
    fn test_is_bounds_invalid_value() {
        let bounds = DMatrix::from_row_slice(3, 2, &[10.0, 0.0, 9.0, 1.0, 8.0, 2.0]);
        assert!(matches!(
            bounds.is_bounds_valid().unwrap_err(),
            ValidationError::InvalidValue
        ));
    }

    #[test]
    fn test_is_reference_ideal_bounds_valid() {
        let reference_ideal = DMatrix::from_row_slice(3, 2, &[5.0, 7.0, 6.0, 8.0, 7.0, 9.0]);
        let bounds = DMatrix::from_row_slice(3, 2, &[0.0, 10.0, 5.0, 9.0, 6.9, 11.0]);
        assert!(reference_ideal
            .is_reference_ideal_bounds_valid(&bounds)
            .is_ok());
    }

    #[test]
    fn test_is_reference_ideal_bounds_invalid_shape() {
        let reference_ideal = DMatrix::from_row_slice(3, 2, &[5.0, 7.0, 6.0, 8.0, 7.0, 9.0]);
        let bounds = DMatrix::from_row_slice(3, 3, &[0.0, 10.0, 5.0, 1.0, 9.0, 5.0, 2.0, 8.0, 5.0]);
        assert!(matches!(
            reference_ideal
                .is_reference_ideal_bounds_valid(&bounds)
                .unwrap_err(),
            ValidationError::DimensionMismatch
        ));
    }

    #[test]
    fn test_is_reference_ideal_bounds_out_of_bounds() {
        let reference_ideal = DMatrix::from_row_slice(3, 2, &[5.0, 7.0, 6.0, 8.0, 7.0, 9.0]);
        let bounds = DMatrix::from_row_slice(3, 2, &[0.0, 6.0, 1.0, 7.0, 2.0, 8.0]);
        assert!(matches!(
            reference_ideal
                .is_reference_ideal_bounds_valid(&bounds)
                .unwrap_err(),
            ValidationError::InvalidValue
        ));
    }

    #[test]
    fn test_is_within_bounds() {
        let decision_matrix = DMatrix::from_row_slice(3, 2, &[5.0, 7.0, 6.0, 8.0, 7.0, 9.0]);
        let bounds = DMatrix::from_row_slice(2, 2, &[0.0, 10.0, 1.0, 9.0]);
        assert!(decision_matrix.is_within_bounds(&bounds).is_ok());
    }

    #[test]
    fn test_is_within_bounds_invalid_shape() {
        let decision_matrix = DMatrix::from_row_slice(3, 2, &[5.0, 7.0, 6.0, 8.0, 7.0, 9.0]);
        let bounds = DMatrix::from_row_slice(3, 2, &[0.0, 10.0, 1.0, 9.0, 2.0, 8.0]);
        assert!(matches!(
            decision_matrix.is_within_bounds(&bounds).unwrap_err(),
            ValidationError::InvalidShape
        ));
    }

    #[test]
    fn test_is_within_bounds_out_of_bounds() {
        let decision_matrix = DMatrix::from_row_slice(3, 2, &[5.0, 7.0, 6.0, 8.0, 7.0, 9.0]);
        let bounds = DMatrix::from_row_slice(2, 2, &[0.0, 6.0, 1.0, 7.0]);
        assert!(matches!(
            decision_matrix.is_within_bounds(&bounds).unwrap_err(),
            ValidationError::InvalidValue
        ));
    }

    #[test]
    fn test_is_expected_solution_point_in_bounds() {
        let solution_point = DVector::from_vec(vec![5.0, 7.0, 6.0]);
        let bounds = DMatrix::from_row_slice(3, 2, &[0.0, 10.0, 1.0, 9.0, 2.0, 8.0]);
        assert!(solution_point
            .is_expected_solution_point_in_bounds(&bounds)
            .is_ok());
    }

    #[test]
    fn test_is_expected_solution_point_in_bounds_invalid_shape() {
        let solution_point = DVector::from_vec(vec![5.0, 7.0, 6.0]);
        let bounds = DMatrix::from_row_slice(3, 3, &[0.0, 10.0, 5.0, 1.0, 9.0, 5.0, 2.0, 8.0, 5.0]);
        assert!(matches!(
            solution_point
                .is_expected_solution_point_in_bounds(&bounds)
                .unwrap_err(),
            ValidationError::InvalidShape
        ));
    }

    #[test]
    fn test_is_expected_solution_point_in_bounds_out_of_bounds() {
        let solution_point = DVector::from_vec(vec![7.0]);
        let bounds = DMatrix::from_row_slice(1, 2, &[0.0, 6.0]);
        assert!(matches!(
            solution_point
                .is_expected_solution_point_in_bounds(&bounds)
                .unwrap_err(),
            ValidationError::InvalidValue
        ));
    }
}
