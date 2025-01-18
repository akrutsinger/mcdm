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

mod ervd_tests {
    use super::*;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            80.0, 70.0, 87.0, 77.0, 76.0, 80.0, 75.0;
            85.0, 65.0, 76.0, 80.0, 75.0, 65.0, 75.0;
            78.0, 90.0, 72.0, 80.0, 85.0, 90.0, 85.0;
            75.0, 84.0, 69.0, 85.0, 65.0, 65.0, 70.0;
            84.0, 67.0, 60.0, 75.0, 85.0, 75.0, 80.0;
            85.0, 78.0, 82.0, 81.0, 79.0, 80.0, 80.0;
            77.0, 83.0, 74.0, 70.0, 71.0, 65.0, 70.0;
            78.0, 82.0, 72.0, 80.0, 78.0, 70.0, 60.0;
            85.0, 90.0, 80.0, 88.0, 90.0, 80.0, 85.0;
            89.0, 75.0, 79.0, 67.0, 77.0, 70.0, 75.0;
            65.0, 55.0, 68.0, 62.0, 70.0, 50.0, 60.0;
            70.0, 64.0, 65.0, 65.0, 60.0, 60.0, 65.0;
            95.0, 80.0, 70.0, 75.0, 70.0, 75.0, 75.0;
            70.0, 80.0, 79.0, 80.0, 85.0, 80.0, 70.0;
            60.0, 78.0, 87.0, 70.0, 66.0, 70.0, 65.0;
            92.0, 85.0, 88.0, 90.0, 85.0, 90.0, 95.0;
            86.0, 87.0, 80.0, 70.0, 72.0, 80.0, 85.0;
        ];
        let weights = dvector![0.066, 0.196, 0.066, 0.130, 0.130, 0.216, 0.196];
        let criteria_types = mcdm::CriteriaType::profits(weights.len());
        let reference_point = dvector![80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0];
        let ranking = matrix.rank_ervd(&criteria_types, &weights, &reference_point, 2.25, 0.88)?;
        let expected = dvector![
            0.6601578744801836,
            0.5027110047442204,
            0.8847463896075456,
            0.5211495079062973,
            0.6096529567767885,
            0.7957970491733928,
            0.4978130505132295,
            0.5486570777965148,
            0.9079407590474731,
            0.5649609972974899,
            0.0698209896816763,
            0.1987226220443113,
            0.631746193377551,
            0.7159012609089653,
            0.4381108721234074,
            0.9716950549406589,
            0.7670370058345157
        ];
        assert_relative_eq!(ranking, expected, epsilon = 1e-5);

        Ok(())
    }

    #[test]
    fn test_rank_with_cost_criteria() -> Result<(), McdmError> {
        let matrix = dmatrix![
            80.0, 70.0, 87.0, 77.0, 76.0, 80.0, 75.0;
            85.0, 65.0, 76.0, 80.0, 75.0, 65.0, 75.0;
            78.0, 90.0, 72.0, 80.0, 85.0, 90.0, 85.0;
            75.0, 84.0, 69.0, 85.0, 65.0, 65.0, 70.0;
            84.0, 67.0, 60.0, 75.0, 85.0, 75.0, 80.0;
            85.0, 78.0, 82.0, 81.0, 79.0, 80.0, 80.0;
            77.0, 83.0, 74.0, 70.0, 71.0, 65.0, 70.0;
            78.0, 82.0, 72.0, 80.0, 78.0, 70.0, 60.0;
            85.0, 90.0, 80.0, 88.0, 90.0, 80.0, 85.0;
            89.0, 75.0, 79.0, 67.0, 77.0, 70.0, 75.0;
            65.0, 55.0, 68.0, 62.0, 70.0, 50.0, 60.0;
            70.0, 64.0, 65.0, 65.0, 60.0, 60.0, 65.0;
            95.0, 80.0, 70.0, 75.0, 70.0, 75.0, 75.0;
            70.0, 80.0, 79.0, 80.0, 85.0, 80.0, 70.0;
            60.0, 78.0, 87.0, 70.0, 66.0, 70.0, 65.0;
            92.0, 85.0, 88.0, 90.0, 85.0, 90.0, 95.0;
            86.0, 87.0, 80.0, 70.0, 72.0, 80.0, 85.0;
        ];
        let weights = dvector![0.066, 0.196, 0.066, 0.130, 0.130, 0.216, 0.196];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, -1, 1, -1, 1, -1])?;
        let reference_point = dvector![80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0];
        let ranking = matrix.rank_ervd(&criteria_types, &weights, &reference_point, 2.25, 0.88)?;
        let expected = dvector![
            0.6559997559670003,
            0.5217977925957993,
            0.7594602276018567,
            0.7313453836161443,
            0.5452375511596929,
            0.7100620905306492,
            0.6409153841787564,
            0.7463398467002611,
            0.6862817392737902,
            0.5463937478471941,
            0.3540287150358768,
            0.5017106312426843,
            0.6832426474880466,
            0.7646456165917208,
            0.6645042283380466,
            0.6279805495737201,
            0.6745200840907036
        ];
        assert_relative_eq!(ranking, expected, epsilon = 1e-5);

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

mod mairca_tests {
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
        let ranking = normalized_matrix.rank_mairca(&weights)?;
        assert_relative_eq!(
            ranking,
            dvector![0.18125122, 0.27884615, 0.0],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod marcos_tests {
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
        let ranking = matrix.rank_marcos(&criteria_types, &weights)?;
        assert_relative_eq!(
            ranking,
            dvector![0.51306940, 0.36312213, 0.91249658],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod moora_tests {
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
        let ranking = matrix.rank_moora(&criteria_types, &weights)?;
        assert_relative_eq!(
            ranking,
            dvector![-0.12902028, -0.14965973, 0.26377149],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod ocra_tests {
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
        let ranking = matrix.rank_ocra(&criteria_types, &weights)?;
        assert_relative_eq!(
            ranking,
            dvector![0.0, 0.73175174, 3.64463555],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod probid_tests {
    use super::*;

    #[test]
    fn test_rank_with_odd_num_alternatives() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let ranking = matrix.rank_probid(&criteria_types, &weights, false)?;
        assert_relative_eq!(
            ranking,
            dvector![0.31041914, 0.39049427, 1.1111652],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_rank_with_even_num_alternatives() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69;
            1.3, 1.48, 1.2, 0.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let ranking = matrix.rank_probid(&criteria_types, &weights, false)?;
        assert_relative_eq!(
            ranking,
            dvector![0.39286123, 0.40670392, 1.04890283, 0.84786888],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_simple_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let ranking = matrix.rank_probid(&criteria_types, &weights, true)?;
        assert_relative_eq!(
            ranking,
            dvector![0.0, 1.31254211, 3.3648893],
            epsilon = 1e-5
        );

        Ok(())
    }

    #[test]
    fn test_simple_rank_with_more_than_four_alts() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69;
            1.3, 1.48, 1.2, 0.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let ranking = matrix.rank_probid(&criteria_types, &weights, true)?;
        assert_relative_eq!(
            ranking,
            dvector![0.33826075, 0.52928193, 3.95966538, 1.77215771],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod ram_tests {
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
        let ranking = matrix.rank_ram(&criteria_types, &weights)?;
        assert_relative_eq!(
            ranking,
            dvector![1.40671879, 1.39992479, 1.48267751],
            epsilon = 1e-5
        );

        Ok(())
    }
}

mod spotis_tests {
    use super::*;

    #[test]
    fn test_rank() -> Result<(), McdmError> {
        let matrix = dmatrix![
            2.9, 2.31, 0.56, 1.89;
            1.2, 1.34, 0.21, 2.48;
            0.3, 2.48, 1.75, 1.69
        ];
        let weights = dvector![0.25, 0.25, 0.25, 0.25];
        let bounds = dmatrix![0.1, 3.0; 1.0, 2.48; 0.0, 1.9; 0.69, 3.1];
        let criteria_types = mcdm::CriteriaType::from(vec![-1, 1, 1, -1])?;
        let ranking = matrix.rank_spotis(&criteria_types, &weights, &bounds)?;
        assert_relative_eq!(
            ranking,
            dvector![0.57089264, 0.69544822, 0.14071266],
            epsilon = 1e-5
        );

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
