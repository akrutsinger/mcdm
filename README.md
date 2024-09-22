# 🤔 mcdm

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
| Entropy			 | Weights each criterion based on the entropy of all alternatives for the given criterion. |
| Equal              | Assumes all criteria are of equal importance. Each criterion is assigned the same weight. |
| Standard Deviation | Criteria weights are derived from standard deviation across the criteria. This method leads to smaller weights for criteria that have similiar values.
|  |  |

### 📈 Ranking Methods

Ranking algorithms combine normalized data and weights to determine the best alternative. These methods aim to provide a clear ranking of alternatives based on the decision-maker's preferences.

| 🥇 Name               | 📝 Description |
| :----------------: | :---------: |
| Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)          |   Conceptually says the best alternative should have the shortest geometric distance from the postivei ideal solution and the longest distance form the negative ideal solution.  |
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
    errors::McdmError, methods::TOPSIS, normalization::MinMax,
    weights::Equal,
};
use ndarray::{array, Array1, Array2};

fn main() -> Result<(), McdmError> {
    // Define the decision matrix (alternatives x criteria)
    let alternatives: Array2<f64> = array![[4.0, 7.0, 8.0], [2.0, 9.0, 6.0], [3.0, 6.0, 9.0]];
    let criteria_types: Array1<i8> = array![-1, 1, 1];
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

Distributed under the MIT license. See [LICENSE-MIT.txt](LICENSE-MIT.txt) for more information.