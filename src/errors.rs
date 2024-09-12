use error_set::error_set;

error_set! {
    /// Unified error type for the entire `mcdm` crate.
    ///
    /// This enum encapsulates all possible error types that may arise when using the various
    /// modules of the `mcdm` crate, such as during normalization, weighting, or ranking operations.
    McdmError = MethodError || NormalizationError || WeightingError;

    /// Error type for method-specific failures.
    ///
    /// This variant of `McdmError` captures errors specific to decision-making methods such as
    /// TOPSIS or other ranking algorithms. Currently, it includes general placeholders but can be
    /// extended to provide more precise errors as method implementations evolve.
    MethodError = {
        /// A general error placeholder. This should be replaced with more specific error types as
        /// the library grows.
        AnError,
    } || NormalizationError;

    /// Error type for normalization-related failures.
    ///
    /// This variant captures all the errors that may occur during the normalization of a decision
    /// matrix. These errors may include issues such as dimension mismatches, incorrect criteria
    /// types, or invalid ranges.
    NormalizationError = {
        /// Thrown when the dimensions of the decision matrix do not match the expected dimensions
        /// (e.g., criteria types array has a different length than the number of columns in the matrix).
        DimensionMismatch,
        /// Thrown when the criteria types array contains invalid values (i.e., anything other than
        /// -1 for cost or 1 for profit).
        NormalizationCriteraTypeMismatch,
        /// Thrown when an empty matrix is provided for normalization.
        EmptyMatrix,
        /// Thrown when the minimum value of a criterion cannot be determined (e.g., for certain criteria).
        NoMinimum,
        /// Thrown when the maximum value of a criterion cannot be determined.
        NoMaximum,
        /// Thrown when all values of a criterion are the same, resulting in a zero range.
        /// Effectively a division by zero error.
        ZeroRange,
    } || ValidationError;

    /// Error type for weighting-related failures.
    ///
    /// This variant captures errors encountered during the calculation of weights for the decision
    /// matrix criteria. These errors may involve invalid matrix dimensions, lack of minimum/maximum
    /// values, or incorrect weight ranges.
    WeightingError = {
        /// Thrown when the dimensions of the weight vector or decision matrix do not match.
        DimensionMismatch,
        /// Thrown when an empty matrix is provided for weight calculation.
        EmptyMatrix,
        /// Thrown when no minimum value for a criterion can be found, making it impossible to
        /// compute weights.
        NoMinimum,
        /// Thrown when no maximum value for a criterion can be found.
        NoMaximum,
        /// Thrown when all values of a criterion are equal, leading to undefined weight calculations.
        ZeroRange,
    };

    /// Error type for validation failures.
    ///
    /// This variant is used to capture validation-related errors that occur during checks of input
    /// data, such as invalid values provided by the user.
    ValidationError = {
        /// Thrown when an input value is invalid (e.g., a criterion type that is neither -1 nor 1).
        InvalidValue,
    };
}
