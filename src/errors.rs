use error_set::error_set;

error_set! {
    /// Unified error type for the entire `mcdm` crate.
    ///
    /// This enum encapsulates all possible error types that may arise when using the various
    /// modules of the `mcdm` crate, such as during normalization, weighting, or ranking operations.
    McdmError = MethodError;

    /// Error type for method-specific failures.
    ///
    /// This variant of `McdmError` captures errors specific to decision-making methods such as
    /// TOPSIS or other ranking algorithms. Currently, it includes general placeholders but can be
    /// extended to provide more precise errors as method implementations evolve.
    MethodError = {
        /// A general error placeholder. This should be replaced with more specific error types as
        /// the library grows.
        AnError,
    } || ValidationError;

    /// Error type for validation failures.
    ///
    /// This variant of `McdmError` captures validation-related errors that occur during checks of
    /// input data, such as invalid values provided by the user. This variant is used for weighting
    /// and normalization functions.
    ValidationError = {
        /// Dimensions of the weight vector, normalization, or decision matrix do not match.
        DimensionMismatch,
        /// Empty matrix is provided for weight calculation.
        EmptyMatrix,
        /// Input value is invalid (e.g., a criterion type that is neither -1 nor 1).
        InvalidValue,
        /// Criteria types array contains invalid values (i.e., anything other than
        /// -1 for cost or 1 for profit).
        NormalizationCriteraTypeMismatch,
        /// No minimum value for a criterion can be found, making it impossible to
        /// compute weights.
        NoMinimum,
        /// No maximum value for a criterion can be found.
        NoMaximum,
        /// Value of zero is improperly included in a calculation, leading to possible undefined
        /// calculations.
        ZeroRange,
    };
}
