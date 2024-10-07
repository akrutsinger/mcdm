mod criteria_type_tests {
    use mcdm::errors::ValidationError;
    use mcdm::CriteriaType;

    #[test]
    fn test_from_valid_values() {
        let input = vec![-1, 1, -1, 1];
        let expected = vec![
            CriteriaType::Cost,
            CriteriaType::Profit,
            CriteriaType::Cost,
            CriteriaType::Profit,
        ];
        let result = CriteriaType::from(input);
        assert_eq!(result.unwrap(), expected);
    }

    #[test]
    fn test_from_invalid_values() {
        let input = vec![-1, 2, 1];
        let result = CriteriaType::from(input);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ValidationError::InvalidValue));
    }

    #[test]
    fn test_profits_length() {
        let len = 5;
        let result = CriteriaType::profits(len);
        assert_eq!(result.len(), len);
        assert!(result.iter().all(|x| *x == CriteriaType::Profit));
    }

    #[test]
    fn test_type_switch() {
        let input = vec![
            CriteriaType::Cost,
            CriteriaType::Profit,
            CriteriaType::Profit,
        ];
        let result = CriteriaType::switch(input);
        assert_eq!(
            result,
            vec![CriteriaType::Profit, CriteriaType::Cost, CriteriaType::Cost]
        );
    }
}
