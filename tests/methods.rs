use approx::assert_abs_diff_eq;
use mcdm::methods::*;
use ndarray::array;

mod topsis_tests {
    use super::*;

    #[test]
    fn test_rank() {
        let matrix = array![
            [0.1, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3, 0.3]
        ];
        let weights = array![0.25, 0.25, 0.25, 0.25];
        let ranking = TOPSIS::rank(&matrix, &weights).unwrap();
        assert_abs_diff_eq!(ranking, array![0.0, 0.5, 1.0], epsilon = 1e-5);
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
