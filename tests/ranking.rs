use approx::assert_relative_eq;
use mcdm::errors::McdmError;
use mcdm::normalization::Normalize;
use mcdm::ranking::Rank;
use nalgebra::{dmatrix, dvector};

mod aras_tests {
    use super::*;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let ranking = matrix.rank_aras(&criteria_types, &weights)?;
        assert_relative_eq!(
            ranking,
            dvector![0.49447117, 0.35767527, 1.0],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod cocoso_tests {
    use super::*;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_min_max(&criteria_types)?;
        let ranking = normalized_matrix.rank_cocoso(&weights)?;
        assert_relative_eq!(
            ranking,
            dvector![3.24754746, 1.14396494, 5.83576765],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_with_zero_weights() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.0, 0.0];
        let ranking = matrix.rank_cocoso(&weights)?;

        assert_eq!(ranking.iter().all(|&x| x.is_nan()), true);

        Ok(())
    }

    #[test]
    fn test_rank_with_invalid_dimensions() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.6];
        let ranking = matrix.rank_cocoso(&weights);
        assert!(ranking.is_err());

        Ok(())
    }
}

mod codas_tests {
    use super::*;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_linear(&criteria_types)?;
        let ranking = normalized_matrix.rank_codas(&weights, 0.02)?;
        assert_relative_eq!(
            ranking,
            dvector![-0.40977725, -1.15891275, 1.56869],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod copras_tests {
    use super::*;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let ranking = matrix.rank_copras(&criteria_types, &weights)?;
        assert_relative_eq!(
            ranking,
            dvector![1.0, 0.6266752, 0.92104753],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod edas_tests {
    use super::*;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let ranking = matrix.rank_edas(&criteria_types, &weights)?;
        assert_relative_eq!(
            ranking,
            dvector![0.04747397, 0.04029913, 1.0],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod mabac_tests {
    use super::*;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_min_max(&criteria_types)?;
        let ranking = normalized_matrix.rank_mabac(&weights)?;
        assert_relative_eq!(
            ranking,
            dvector![-0.01955314, -0.31233795, 0.52420052],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_with_zero_weights() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.0, 0.0];
        let ranking = matrix.rank_mabac(&weights)?;
        assert_eq!(ranking, dvector![0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_rank_with_invalid_dimensions() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.6];
        let ranking = matrix.rank_mabac(&weights);
        assert!(ranking.is_err());

        Ok(())
    }
}

mod topsis_tests {
    use super::*;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_min_max(&criteria_types)?;
        let ranking = normalized_matrix.rank_topsis(&weights)?;
        assert_relative_eq!(
            ranking,
            dvector![0.52910451, 0.72983217, 0.0],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_with_zero_weights() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.0, 0.0];
        let ranking = matrix.rank_topsis(&weights);
        assert!(ranking.is_err());

        Ok(())
    }

    #[test]
    fn test_rank_with_invalid_dimensions() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.6];
        let ranking = matrix.rank_topsis(&weights);
        assert!(ranking.is_err());

        Ok(())
    }
}

mod weighted_product_tests {
    use super::*;
    use mcdm::CriteriaType;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_type = CriteriaType::from(vec![-1, 1, 1, -1])?;
        let normalized_matrix = matrix.normalize_sum(&criteria_type)?;
        let ranking = normalized_matrix.rank_weighted_product(&weights)?;
        assert_relative_eq!(
            ranking,
            dvector![0.21711531, 0.17273414, 0.53281425],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_rank_with_invalid_dimensions() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.6];
        let ranking = matrix.rank_weighted_product(&weights);
        assert!(ranking.is_err());

        Ok(())
    }

    #[test]
    fn test_rank_with_zero_weights() -> Result<(), McdmError> {
        let matrix = dmatrix![0.3, 0.5; 0.6, 0.9; 0.1, 0.2];
        let weights = dvector![0.0, 0.0];
        let criteria_type = CriteriaType::from(vec![1, -1])?;
        let normalized_matrix = matrix.normalize_sum(&criteria_type)?;
        let ranking = normalized_matrix.rank_weighted_product(&weights)?;
        assert_eq!(ranking, dvector![1.0, 1.0, 1.0]);

        Ok(())
    }
}

mod weighted_sum_tests {
    use super::*;
    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.6, 0.4];
        let ranking = matrix.rank_weighted_sum(&weights)?;
        assert_relative_eq!(ranking, dvector![0.44, 0.5, 0.58], epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_with_zero_weights() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.0, 0.0];
        let ranking = matrix.rank_weighted_sum(&weights)?;
        assert_eq!(ranking, dvector![0.0, 0.0, 0.0]);

        Ok(())
    }

    #[test]
    fn test_rank_with_invalid_dimensions() -> Result<(), McdmError> {
        let matrix = dmatrix![0.2, 0.8; 0.5, 0.5; 0.9, 0.1];
        let weights = dvector![0.6];
        let ranking = matrix.rank_weighted_sum(&weights);
        assert!(ranking.is_err());

        Ok(())
    }
}
