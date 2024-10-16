# 🤔 mcdm

[![Build Status](https://github.com/akrutsinger/mcdm/actions/workflows/rust.yml/badge.svg)](https://github.com/akrutsinger/mcdm/actions)
[![Crates.io](https://img.shields.io/crates/v/mcdm.svg?logo=rust)](https://crates.io/crates/mcdm)
[![Documentation](https://img.shields.io/docsrs/mcdm?logo=docs.rs
)](https://docs.rs/mcdm)

`mcdm` is a Rust library to assist with solving Multiple-Criteria Decision Making (MCDM) problems. It provides a comprehensive set of tools for decision analysis, including normalization techniques, weighting methods, and ranking algorithms.

## ✨ Features

MCDM involves evaluating multiple conflicting criteria when making decisions. This library aims to provide various techniques to normalize data, assign weights to criteria, and rank alternatives. Below is a list of the components that have been implemented.

### 🔄 Normalization

Normalization ensures that criteria with different units of measurement can be compared on a common scale. Each method has a specific use case depending on the decision problem.

| 🌟 Name               | 📝 Description |
| :----------------: | :---------: |
| Enhanced Accuracy  | Technique proposed by Zeng and Yang in 2013 incorporating criterion's minimums and maximums into the computation. |
| Linear 			 | Similiar to Max normalization, where profit criteria depend on the critions maximum value and cost criteria depend on criterion minimum value. |
| Logarithmic        | Uses natural logarithm in the normalization. |
| Min-Max | Scales the values of each criterion between 0 and 1 (or another range) by using the minimum and maximum values of that criterion. |
| Max                | Similiar to MinMax, but here each element is divided by the maximum value in the column |
| Nonlinear | Similiar to linear normalization, but relies on exponentiation of the criteria to help capture more complexitities in the criteria or when data distributions are skewed. |
| Sum				 | Uses the sum of each alternatives criterion. |
| Vector             | Considers the root of the sum of squares for each criterion. |
| Zavadskas-Turskis  | Convert different criteria into comparable units. |
|  |  |

### ⚖️ Weighting

Weights reflect the relative importance of each criterion. Different weighting techniques help balance criteria appropriately in the decision-making process.

| 📊 Name               | 📝 Description |
| :----------------: | :---------: |
| Angular | Measures angle between each criterion and an ideal point in a multidimensional space. Criteria that are closer to that ideal point (i.e., have smaller angles) are considered more important and assigned higher weights. |
| CRiteria Importance Through Intercriteria Correlation (CRITIC) |Determines weights by combining the variability (standard deviation) of each criterion and the correlation between criteria. A criterion with high variablity and low correlation with others is considered more important. |
| Entropy			 | Measures the uncertainty or disorder within the criterion's values across alternatives. Criteria with higher variability (i.e., more dispersed values) have higher entropy and are assigned lower weights. Criteria with more structured values receive higher weights. |
| Equal              | Assumes all criteria are equally important and each criterion is assigned the same weight. |
| Gini | Based on the Gini coefficient, which measures inequality or dispersion. Criteria with greaterinequality or larger variation across alternatives receive higher weights. |
| Method Based on the Removal Effects of Criiteria (MEREC) | Evaluates how the absence of a criterion affects the overall decision. Criteria that, when removed, significantly change the ranking of alternatives are considered more important and are given higher weights. |
| Standard Deviation | Criteria weights are derived from standard deviation of criteria across alternatives. Criteria with hiigher standard deviation (i.e., more variation) are given higher weights because they better differentiate the alternatives.
| Variance | Similiar to standard deviation, variance weighting assigns weights to criteria based on the dispersion (variance) of their values across alternatives. Criter with higher variance are consideredmore important. |

### 📈 Ranking Methods

Ranking algorithms combine normalized data and weights to determine the best alternative. These methods aim to provide a clear ranking of alternatives based on the decision-maker's preferences.

| 🥇 Name               | 📝 Description |
| :----------------: | :---------: |
| COmbined COmpromise SOlution (COCOSO) | Combines aspects of compromise ranking methods to evaluate and rank alternatives by integrating three approaches: simple additive weighting (SAW), weighted product model (WPM), and the average ranking of alternatives based on their relative performance. This method finds a compromise solution by blending these different ranking strategies, making it robust in handling conflicting criteria. |
| Multi-Attributive Border Approximation Area Comparison (MABAC) | A compensatory ranking method that allows for the trade-off between criteria. The general approach is to calculate the distance between each alternative and an ideal solution called the "border approximation area". |
| Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)          |   Conceptually says the best alternative should have the shortest geometric distance from the postivei ideal solution and the longest distance form the negative ideal solution.  |
| Weighted Product Model | Ranks alternatives based on product of each weighted alternative. |
| Weighted Sum | Ranks alternatives based on the weighted sum of the alternatives criteria values. |
|  |  |

## 🚀 Usage

To start using `mcdm`, add this to your `Cargo.toml`:

```toml
[dependencies]
mcdm = "0.1"
```

### 🧪 Example

Here’s an example demonstrating how to use the mcdm library for decision-making with the TOPSIS method:

```rust
use mcdm::{
    errors::McdmError,
    rankings::{Rank, TOPSIS},
    normalization::{MinMax, Normalize},
    weights::{Equal, Weight},
    CriteriaType,
};
use ndarray::array;

fn main() -> Result<(), McdmError> {
    // Define the decision matrix (alternatives x criteria)
    let alternatives = array![[4.0, 7.0, 8.0], [2.0, 9.0, 6.0], [3.0, 6.0, 9.0]];
    let criteria_types = CriteriaType::from_vec(vec![-1, 1, 1])?;
    // Apply normalization using Min-Max
    let normalized_matrix = MinMax::normalize(&alternatives, &criteria_types)?;
    // Alternatively, use equal weights
    let equal_weights = Equal::weight(&normalized_matrix)?;
    // Apply the TOPSIS method for ranking
    let ranking = TOPSIS::rank(&normalized_matrix, &equal_weights)?;
    // Output the ranking
    println!("Ranking: {:.3?}", ranking);
    Ok(())
}
```

## 🤝 Contributing

Contributing is what makes the open source community thrive. Any contributions to enhance `mcdm` are welcome and greatly apprecaited!

If you have any suggestions that could make this project better, please feel free to fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork this repo
2. Create your feature branch (`git checkout -b feature/new-decision-method`)
3. Commit your changes (`git commit -m 'Add new decision method'`)
4. Push to the remote branch (`git push origin feature/new-decision-method`)
5. Open a pull request

## 📄 License

Distributed under the MIT license. See [LICENSE-MIT](LICENSE-MIT) for more information.