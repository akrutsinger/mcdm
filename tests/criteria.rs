mod criteria_type_tests {
    use mcdm::errors::McdmError;
    use mcdm::errors::ValidationError;
    use mcdm::{CriteriaTypes, CriterionType};

    #[test]
    fn test_from_valid_values() -> Result<(), McdmError> {
        let input = &[-1, 1, -1, 1];
        let result = CriteriaTypes::from_slice(input)?;
        assert_eq!(result[0], CriterionType::Cost);
        assert_eq!(result[1], CriterionType::Profit);
        assert_eq!(result[2], CriterionType::Cost);
        assert_eq!(result[3], CriterionType::Profit);

        Ok(())
    }

    #[test]
    fn test_from_invalid_values() -> Result<(), McdmError> {
        let input = &[-1, 2, 1];
        let result = CriteriaTypes::from_slice(input);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ValidationError::InvalidValue));

        Ok(())
    }

    #[test]
    fn test_profits_length() {
        let len = 5;
        let result = CriteriaTypes::all_profits(len);
        assert_eq!(result.len(), len);
        assert!(result.iter().all(|x| *x == CriterionType::Profit));
    }

    #[test]
    fn test_type_switch() -> Result<(), McdmError> {
        let input = CriteriaTypes::from_slice(&[-1, 1, -1])?;
        let result = CriteriaTypes::switch(&input);
        assert_eq!(result[0], CriterionType::Profit);
        assert_eq!(result[1], CriterionType::Cost);
        assert_eq!(result[2], CriterionType::Profit);

        Ok(())
    }

    #[test]
    fn test_criterion_type_conversion() {
        assert_eq!(CriterionType::try_from(-1).unwrap(), CriterionType::Cost);
        assert_eq!(CriterionType::try_from(1).unwrap(), CriterionType::Profit);
        assert!(CriterionType::try_from(0).is_err());
    }

    #[test]
    fn test_criteria_types_from_ints() -> Result<(), McdmError> {
        let criteria = CriteriaTypes::from_slice(&[-1, 1, -1])?;
        assert_eq!(criteria.len(), 3);
        assert_eq!(criteria[0], CriterionType::Cost);
        assert_eq!(criteria[1], CriterionType::Profit);
        assert_eq!(criteria[2], CriterionType::Cost);

        Ok(())
    }

    #[test]
    fn test_invalid_criteria_types() {
        assert!(CriteriaTypes::from_slice(&[-1, 0, 1]).is_err());
    }
}
