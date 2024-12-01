use approx::assert_relative_eq;
use mcdm::errors::McdmError;
use mcdm::normalization::Normalize;
use mcdm::weighting::Weight;
use mcdm::CriteriaType;
use nalgebra::{dmatrix, dvector};

mod angular_tests {
    use super::*;

    #[test]
    fn test_weight() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let normalized_matrix = matrix.normalize_sum(&CriteriaType::profits(matrix.ncols()))?;
        let weights = normalized_matrix.weight_angular()?;
        let expected_weights = dvector![0.37183274, 0.14136016, 0.39029729, 0.09650981];
        assert_relative_eq!(weights, expected_weights, epsilon = 1e-5);

        Ok(())
    }
}

mod critic_tests {
    use super::*;

    #[test]
    fn test_weight() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaType::profits(matrix.ncols());
        let normalized_matrix = matrix.normalize_min_max(&criteria_types)?;
        let weights = normalized_matrix.weight_critic()?;
        let expected_weights = dvector![0.22514451, 0.21769531, 0.24375671, 0.31340347];
        assert_relative_eq!(weights, expected_weights, epsilon = 1e-5);

        Ok(())
    }
}

mod entropy_tests {
    use super::*;

    #[test]
    fn test_weight() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaType::profits(matrix.ncols());
        let normalized_matrix = matrix.normalize_sum(&criteria_types)?;
        let weights = normalized_matrix.weight_entropy()?;
        let expected_weights = dvector![0.45009235, 0.05084365, 0.4778931, 0.0211709];
        assert_relative_eq!(weights, expected_weights, epsilon = 1e-5);

        Ok(())
    }
}

mod equal_tests {
    use super::*;

    #[test]
    fn test_weight() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = matrix.weight_equal()?;
        let expected_weights = dvector![0.25, 0.25, 0.25, 0.25];
        assert_relative_eq!(weights, expected_weights, epsilon = 1e-5);

        Ok(())
    }
}

mod gini_tests {
    use super::*;

    #[test]
    fn test_weight() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = matrix.weight_gini()?;
        let expected_weights = dvector![0.38917745, 0.12248175, 0.40248266, 0.08585814];
        assert_relative_eq!(weights, expected_weights, epsilon = 1e-5);

        Ok(())
    }
}

mod merec_tests {
    use super::*;

    #[test]
    fn test_weight() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaType::from(vec![-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_linear(&CriteriaType::switch(criteria_types))?;
        let weights = normalized_matrix.weight_merec()?;
        let expected_weights = dvector![0.40559526, 0.14185966, 0.37609753, 0.07644754];
        assert_relative_eq!(weights, expected_weights, epsilon = 1e-5);

        Ok(())
    }
}

mod standard_deviation_tests {
    use super::*;

    #[test]
    fn test_weight() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = matrix.weight_standard_deviation()?;
        let expected_weights = dvector![0.41871179, 0.19503155, 0.25600523, 0.13025143];
        assert_relative_eq!(weights, expected_weights, epsilon = 1e-5);

        Ok(())
    }
}

mod variance_tests {
    use super::*;

    #[test]
    fn test_weight() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let criteria_types = CriteriaType::profits(matrix.ncols());
        let normalized_matrix = matrix.normalize_min_max(&criteria_types)?;
        let weights = normalized_matrix.weight_variance()?;
        let expected_weights = dvector![0.23572429, 0.26602392, 0.25117527, 0.24707653];
        assert_relative_eq!(weights, expected_weights, epsilon = 1e-5);

        Ok(())
    }
}
