mod criteria_type_tests {
    use mcdm::{CriteriaTypes, CriterionType, McdmError, ValidationError};

    #[test]
    fn test_from_valid_slice() -> Result<(), McdmError> {
        let input = &[-1, 1, -1, 1];
        let result = CriteriaTypes::from_slice(input)?;
        assert_eq!(result[0], CriterionType::Cost);
        assert_eq!(result[1], CriterionType::Profit);
        assert_eq!(result[2], CriterionType::Cost);
        assert_eq!(result[3], CriterionType::Profit);

        Ok(())
    }

    #[test]
    fn test_from_invalid_slice() -> Result<(), McdmError> {
        let input = &[-1, 2, 1];
        let result = CriteriaTypes::from_slice(input);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ValidationError::InvalidValue));

        Ok(())
    }

    #[test]
    fn test_all_profits() {
        let len = 5;
        let result = CriteriaTypes::all_profits(len);
        assert_eq!(result.len(), len);
        assert!(result.iter().all(|x| *x == CriterionType::Profit));
    }

    #[test]
    fn test_all_costs() {
        let len = 5;
        let result = CriteriaTypes::all_costs(len);
        assert_eq!(result.len(), len);
        assert!(result.iter().all(|x| *x == CriterionType::Cost));
    }

    #[test]
    fn test_type_switch() -> Result<(), McdmError> {
        let input = CriteriaTypes::from_slice(&[-1, 1, -1])?;
        let result = CriteriaTypes::invert_types(&input);
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
}
