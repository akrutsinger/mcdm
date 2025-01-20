# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Working towards making the code more idiomatic and reducing unnecessary calculations inside of loops.
- Refactored `CriteriaType` code into its own module.

### Fixed

- Bug fixes

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