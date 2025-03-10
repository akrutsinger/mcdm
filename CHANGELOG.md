# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `MOOSRA` ranking method.
- New crate-only scoped `crate::float::Float` trait for `f64` type. This allows removing reliance on the misused `nalgebra::ComplexField` type while providing access to `libm`'s functions for `f64` arithmetic. Now `mcdm` should actually be `#![no_std]`!

### Changed

- Tried to make the README more clear and provice more useful information.

## [0.4.0] - 2025-02-22

### Added

- More documentation. This includes more details in many of the `ranking`/`weighting`/`normalization` traits as well as documenting functions that were originally neglected from down right laziness.
- Ability for user to provide a preference value to the `COCOSO` ranking method. This value preferences which kind of scoring strategy the `COCOSO` ranking method uses.
- Ability for user to provide a preference or `None` to the `ERVD` and `WASPAS` ranking methods. The the user provides `None`, the ranking methods default to the recommended default values.
- `CILOS` weighting method. Adding this also required adding an additional error variant, `ValidationError::SingularMatrix`, in the event a certain matrix isn't a square matrix when it actually should be.
- `IDOCRIW` weighting method.
- `D-CRITIC` weighting method.
- New `correlation` module for computing correlation coefficients of matrices.
- Support for `#![no_std]`.

### Changed

- Working towards making the code more idiomatic and reducing unnecessary calculations inside of loops.
- Refactored `CriteriaType` code into its own module and type `CriteriaTypes`. Instead of being a `Vec`, `CriteriaTypes` is an alias to `DVector<CriterionType>`, where `CriterionType` is an enum of either `Cost` or `Profit`.
- Renamed many of the ranking and normalization functions `type` parameter to `criteria_types` for more clarity to the user.
- Renamed `CriteriaTypes`'s `switch` function to `invert_types`. Naming is hard, hopefully this is a little more descriptive.
- Renamed `DMatrixExt`'s `scale_columns` function to `apply_column_weights`.
- Re-exported all traits from modules to improve ergonomic feel of using the library.
- All dynamic allocations that used `Vec` now go straight into a DVector from an iterator. This allowed for the transition to `no_std`.
- Instead of using the standard library's f64 ln(), sqrt(), powf(), etc., functions, switched to `nalgebra`'s `ComplexField`'s implementation of ln(), sqrt(), etc., to allow for `no_std` in the library.
- Updated rust edition to 2024.

### Removed

- Removed unused function `get_ideal_from_bounds` in `DMatrixExt`.

### Fixed

- Bug fixes
- Removed many of the pedantic clippy lint warnings.
- Moved the rest of the checks for empty matrix to be before assigning variables on the size of the matrix. 

## [0.3.1] - 2024-01-19

### Fixed

- Documentation had many inconsistencies and blantant errors. Should mostly be fixed, but more could still be lerking.

## [0.3.0] - 2024-01-19

### Added

- Ranking methods:
  - ERVD, MAIRCA, MARCOS, MOORA, OCRA, PROBID, RAM, RIM, SPOTIS, WASPAS
- OCRA-specific normalization method.
- SPOTIS-specific normalization method.

## [0.2.0] - 2024-11-30

### Added

- A few more ranking methods.

### Changed

- All weighting, normalization, and ranking methods are implemented for the `DMatrix<f64>` type.
- Moved from `ndarray` to `nalgebra` for matrix types.
- Renamed `weight_criteria` to `scale_columns` to be a little better descriptive; naming is hard.

## [0.1.6] - 2024-10-29

### Added

- More ranking methods.
- More weighting tests.
- Function named `switch` to `CriteriaTypes`. Useful for weighting methods like `MEREC`.

### Changed

- Renamed modules `weights` to `weighting` and `methods` to `ranking` to be more clear in their function and put them in the singular form.
- Tests now return appropriate errors instead of panicing on failures.

### Fixed

- Entropy and MEREC weighting method documentation to add more clarity.
- Entopy weighting calculation.
- Doc typos and clarity.
- Getting more disciplined at actually trying to better follow semantic versioning.

## [0.1.5] - 2024-10-05

### Added

- More tests (still not enough yet).
- Weighting methods.

### Fixed

- Enhanced Accuracy and Logarithmic normalization methods.
- More ranking methods.

## [0.1.1] - 2024-09-11

### Added

- CHANGELOG.md file.

###  Fixed

- README.md name so [https://crates.io](https://crates.io/) will properly display the README.

## [0.1.0] - 2024-09-11

### Added

- Initial release.