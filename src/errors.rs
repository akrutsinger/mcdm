use error_set::error_set;

error_set! {
    /// Unified error type for the entire `mcdm` crate.
    ///
    /// This enum encapsulates all possible error types that may arise when using the various
    /// modules of the `mcdm` crate, such as during normalization, weighting, or ranking operations.
    McdmError = RankingError || NormalizationError || WeightingError || ValidationError;

    /// Error type for ranking failures.
    ///
    /// This variant of `McdmError` captures errors specific to decision-making methods such as
    /// TOPSIS or other ranking algorithms. Currently, it includes general placeholders but can be
    /// extended to provide more precise errors as method implementations evolve.
    RankingError = {
        /// A general error placeholder for ranking. Replace with more specific error types as the
        /// ranking algorithms evolve.
        AnError,
    } || ValidationError;

    /// Normalization related errors.
    ///
    /// This variant captures all the errors that may occur during the normalization of a decision
    /// matrix. These errors may include issues such as dimension mismatches, incorrect criteria
    /// types, or invalid ranges.
    NormalizationError = {
        /// A general error placeholder for normalization. Replace with more specific errors as
        /// normalization methods are refined.
        AnError,
    } || ValidationError;

    /// Weighting related errors.
    ///
    /// This variant captures errors encountered during the calculation of weights for the decision
    /// matrix criteria. These errors may involve invalid matrix dimensions, lack of minimum/maximum
    /// values, or incorrect weight ranges.
    WeightingError = {
        /// Indicates that the input array was empty.
        EmptyInput(ndarray_stats::errors::EmptyInput),
    } || ValidationError;

    /// Data validation errors.
    ///
    /// This variant captures validation-related errors that occur during checks of input data,
    /// such as invalid matrix dimensions or improperin values provided by the user. VAlidation
    /// errors may occur across normalization, weighting, or ranking functions.
    ValidationError = {
        /// The dimensions of the input matrices (e.g., weight vector, decision matrix) do not match
        /// as expected.
        DimensionMismatch,
        /// An empty matrix was provided where a non-empty matrix was expected (e.g., in weight
        /// calculation).
        EmptyMatrix,
        /// Input contains an invalid value (e.g., a criterion type that is neither -1 nor 1).
        InvalidValue,
        /// The criteria types array contains invalid values (e.g., values other than -1 for cost
        /// criteria or 1 for profit criteria).
        NormalizationCriteraTypeMismatch,
        /// No minimum value for a criterion was found, making it impossible to complete the
        /// calculation (e.g., in weighting or normalization).
        NoMinimum,
        /// No maximum value for a criterion was found, which is necessary for certain calculations.
        NoMaximum,
        // A zero range was encountered where a non-zero range was expected, leading to an
        /// undefined or invalid calculation.
        ZeroRange,
    };
}
