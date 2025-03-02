# 🤔 mcdm

[![Build Status](https://github.com/akrutsinger/mcdm/actions/workflows/rust.yml/badge.svg)](https://github.com/akrutsinger/mcdm/actions)
[![Crates.io](https://img.shields.io/crates/v/mcdm.svg?logo=rust)](https://crates.io/crates/mcdm)
[![Documentation](https://img.shields.io/docsrs/mcdm?logo=docs.rs
)](https://docs.rs/mcdm)

`mcdm` is a Rust library to assist with solving Multiple-Criteria Decision Making (MCDM) problems providing a comprehensive set of tools for decision analysis, including normalization techniques, weighting methods, and ranking algorithms.

## 🚀 Quick Start

Add `mcdm` to your `Cargo.toml`:

```toml
[dependencies]
# TODO: replace the * with the latest version
mcdm = "*"
```

Basic usage example:

```rust
use mcdm::{CriteriaTypes, McdmError, Normalize, Rank, Weight};
use nalgebra::dmatrix;

fn main() -> Result<(), McdmError> {
    // Create decision matrix, apply TOPSIS ranking method
    let decision_matrix = dmatrix![4.0, 7.0; 2.0, 9.0; 3.0, 6.0];
    let criteria = CriteriaTypes::from_slice(&[-1, 1])?; // Minimize first, maximize second

    let normalized = decision_matrix.normalize_min_max(&criteria)?;
    let ranking = normalized.rank_topsis(&normalized.weight_equal()?)?;

    println!("Alternative rankings: {:.3}", ranking);
    Ok(())
}
```

For more comprehensive examples, see the [Examples](#-example) section below.

## ✨ Features

MCDM involves evaluating multiple conflicting criteria when making decisions. This library aims to provide various techniques to normalize data, assign weights to criteria, and rank alternatives. Below is a list of the components that have been implemented.

### 🔄 Normalization

Normalization ensures that criteria with different units of measurement can be compared on a common scale. Each method has a specific use case depending on the decision problem.

| 🌟 Name           | 📝 Description | 🔍 Best For |
| :---------------: | :------------: | :---------: |
| Enhanced Accuracy | Technique proposed by Zeng and Yang in 2013 incorporating criterion's minimums and maximums into the computation. | Complex decision problems with wide ranges of criteria values. |
| Linear 			| Similiar to Max normalization, where profit criteria depend on the critions maximum value and cost criteria depend on criterion minimum value. | When a linear relationship between criteria values is assumed. |
| Logarithmic       | Uses natural logarithm in the normalization. | When data has exponential distribution or high value variations. |
| MARCOS            | Method used by the MARCOS ranking method. |
| Max               | Similiar to MinMax, but here each element is divided by the maximum value in the column | When only upper bounds matter. |
| Min-Max           | Scales the values of each criterion between 0 and 1 (or another range) by using the minimum and maximum values of that criterion. | General purpose, widely applicable. |
| Nonlinear         | Similiar to linear normalization, but relies on exponentiation of the criteria to help capture more complexitities in the criteria or when data distributions are skewed. | Non-linear relationships between criteria. |
| OCRA              | Method used by the OCRA ranking method. | when evaluating operational competitiveness with emphasis on both beneficial and non-beneficial criteria. |
| RIM               | Method used by the RIM ranking method. | When comparing alternatives against reference ideals with boundary constraints. |
| SPOTIS            | Method used by the SPOTIS ranking method. | When stability in preference ordering is critical and distance from ideal solution matters. |
| Sum				| Uses the sum of each alternatives criterion. | When proportional relationships are important. |
| Vector            | Considers the root of the sum of squares for each criterion. | When dealing with multi-dimensional data. |
| Zavadskas-Turskis | Convert different criteria into comparable units. | When criteria have vastly different units. |

### ⚖️ Weighting

Weights reflect the relative importance of each criterion. The choice of weighting method depends on whether you want weights derived objectively from the data or if you have subjective preferences to incorporate.

| 📊 Name  | 📝 Description | 🔍 Best For |
| :------: | :------------: | :---------: |
| Angular | Measures angle between each criterion and an ideal point in a multidimensional space. Criteria that are closer to that ideal point (i.e., have smaller angles) are considered more important and assigned higher weights. | Geometric Interpretation of criteria importance. |
| Criterion Impact LOSs (CILOS) | Assigns weights to criteria based on their impact loss, where the importance of criterion is determined by how much the overall decision performance would decrease if that criterion were removed. This method quantifies the significance of each decision criterion by analyzing its contribution to decision-making, ensuring that more influencial criteria receive higher weights. | When measuring the relative importance of criteria based on their contribution to overall decision outcome. |
| CRiteria Importance Through Intercriteria Correlation (CRITIC) | Determines weights by combining the variability (standard deviation) of each criterion and the Pearson correlation between criteria. A criterion with high variablity and low correlation with others is considered more important. | When you want to acount for both vriation and independence. |
| Distance Correlation-based CRiteria Importance Through Intercriteria Correlation (D-CRITIC) | Determines weights by combining the variability (standard deviation) of each criterion and the distance correlation between criteria. A criterion with high variability and low correlation with others is considered more important. | When dealing with non-linear relationships between criteria where Pearson correlation is insufficient. |
| Entropy			 | Measures the uncertainty or disorder within the criterion's values across alternatives. Criteria with higher variability (i.e., more dispersed values) have higher entropy and are assigned lower weights. Criteria with more structured values receive higher weights. | When information content of criteria matter. |
| Equal              | Assumes all criteria are equally important and each criterion is assigned the same weight. | When no preference information is available or for baseline comparison. |
| Gini | Based on the Gini coefficient, which measures inequality or dispersion. Criteria with greaterinequality or larger variation across alternatives receive higher weights. | When focusing on criteria with uneven distributions. |
Integrated Determination of Objective CRIteria Weights (IDOCRIW) | Assigns weights based on the aggregate weights of the `CILOS` and `Entropy` weighting methods. | When combining entropy-based information content with criteria impact for more robust weighting. |
| Method Based on the Removal Effects of Criiteria (MEREC) | Evaluates how the absence of a criterion affects the overall decision. Criteria that, when removed, significantly change the ranking of alternatives are considered more important and are given higher weights. | When sensitivity analysis is important. |
| Standard Deviation | Criteria weights are derived from standard deviation of criteria across alternatives. Criteria with hiigher standard deviation (i.e., more variation) are given higher weights because they better differentiate the alternatives. | When differentiation power matters. |
| Variance | Similiar to standard deviation, variance weighting assigns weights to criteria based on the dispersion (variance) of their values across alternatives. Criter with higher variance are consideredmore important. | When emphasizing criteria with larger data spreads. |

### 📈 Ranking Methods

Ranking algorithms combine normalized data and weights to determine the more preferable alternative. Different methods use different mathematical approaches and may be more suitable depending on the specific decision context.

| 🥇 Name               | 📝 Description | 🔍 Best For |
| :----------------: | :---------: | :---------: |
| Additive Ratio ASsessment (ARAS) | Assesses alternatives by comparing their overall performance to the ideal (best) alternative. The assessment is made by taking the ratio of the normalized and weighted decision matrix to the normalized and weighted "best case" alternatives. | When comparing to an ideal solution is important. |
| COmbined COmpromise SOlution (COCOSO) | Combines aspects of compromise ranking methods to evaluate and rank alternatives by integrating three approaches: simple additive weighting (SAW), weighted product model (WPM), and the average ranking of alternatives based on their relative performance. This method finds a compromise solution by blending these different ranking strategies, making it robust in handling conflicting criteria. | When you want to balance multiple ranking approaches. |
| COmbined Distance-based ASessment (CODAS) | Ranks alternatives based on euclidean distance and taxicab distance from the negative ideal solution. | When different distance measures provide complementary insights. |
| COmplex PRoportional ASsessment (COPRAS) | Ranks alternatives by separately considering the effects of maximizing (beneficial) and minimizing (non-beneficial) index values of attributes. | When beneficial and non-beneficial criteria need separate treatment. |
| Evaluation based on Distance from Average Solution (EDAS) | Ranks alternatives based on how alternatives are evaluated with respct to their distance from the mean solution. | When deviation from average is a key consideration. |
| Election based on Relative Value Distances (ERVD) | Ranks alternatives based on the distance each alternative is from a value decision which represents a positive or negative ideal solution. | When comparing to both positive and negative ideal solutions. |
| Multi-Attributive Border Approximation Area Comparison (MABAC) | A compensatory ranking method that allows for the trade-off between criteria. The general approach is to calculate the distance between each alternative and an ideal solution called the "border approximation area". | When border approximation areas are meaningful. |
| Multi-Attributive Ideal-Real Comparative Analysis (MAIRCA) | Ranking method designed to rank alternatives based on their relative closeness to an ideal solution and distance form a real comparative solution. | When comparing ideal and real solutions. |
| Measurement of Alternatives and Ranking according to COmpromise Solutions (MARCOS) | Designed to rank alternatives of a non-normalized decision matrix based on their performance across multiple criteria. It combines normalization, ideal solutions, and compromise approaches to evaulate alternatives. | Complex decision problems with compromise solutions. |
| Multi-Objective Optimization on the basis of Ratio Analysis (MOORA) | Ranking method that ranks alternatives based on multiple conflicting criteria. | When ratio-based analysis is important. |
| Multi-Objective Optimization on the basis of Simple Ration Analysis (MOOSRA) | Ranking method that ranks alternatives based on multiple conflicting criteria. | When ratio-based analysis is important. |
| Operational Competitiveness Rating Analysis (OCRA) | The OCRA ranking method operates on a non-normalized decision matrix. This method ranks alternatives by comparing their performance across multiple criteria. It is designed to handle both beneficial critieria (those to be maximized, such as profits or quality) and non-beneficial criteria (those to be minimized, such as costs or environmental impact). | When both performance and resource consumption are important. |
| Preference Ranking on the Basis of Ideal-Average Distance (PROBID) | Ranks alternatives based on their proximity to an ideal solution while considering average performance of alternatives. | When balancing ideal solution approach with average performance as a reference point. |
| Root Assessment Method (RAM) | Ranks alternatives by comparing them to a root or reference alternative. RAM considers both the relative performance of alternatives and their deviation from the reference point, ensuring a balanced evaluation across multiple criteria. | When a reference or root alternative is available for comparison. |
| Reference Ideal Method (RIM) | Ranks alternatives using criteria bounds and comparison of the alternatives to a reference ideal for evaluating alternatives. | When explicit bounds for criteria values are known and referenc epoints matter. |
| Stable Preference Ordering Towards Ideal Solution (SPOTIS) | Ranks alternatives based on their distance from an ideal solution | When stability in preference ordering is important with changing criteria values. |
| Technique for Order of Preference by Similarity to Ideal Solution (TOPSIS)          |   Conceptually says the best alternative should have the shortest geometric distance from the postivei ideal solution and the longest distance form the negative ideal solution.  | When both positive and negative ideal solutions are considered. |
| Weighted Aggregated Sum Product ASessment (WASPAS) | Ranks alternatives using both the Weighted Product and Weighted Sum models. WASPAS uses a preferance value, lambda, to give more preferance towards either the Weighted Product or Weighted Sum models. Preferances can be equal towards both models as well. | When combining additive and multiplicative approaches for more stable and balanced rankings. |
| Weighted Product Model | Ranks alternatives based on product of each weighted alternative. | When multiplication better represents the relationship between criteria. |
| Weighted Sum | Ranks alternatives based on the weighted sum of the alternatives criteria values. | Simple problems with straightforward additive criteria. |

### 📊  Examples

### Basic Example: TOPSIS Method
This example demonstrates how to use the TOPSIS methodto rank alternatives:

```rust
use mcdm::{McdmError, Normalize, Rank, Weight, CriteriaTypes};
use nalgebra::dmatrix;

fn main() -> Result<(), McdmError> {
    // Define the decision matrix (alternatives x criteria)
    let alternatives = dmatrix![
        4.0, 7.0, 8.0;  // Alternative A
        2.0, 9.0, 6.0;  // Alternative B
        3.0, 6.0, 9.0   // Alternative C
    ];
    
    // Define the criteria types: 1 for benefit (maximize), -1 for cost (minimize)
    let criteria_types = CriteriaTypes::from_slice(&[-1, 1, 1])?;

    // Step 1: Normalize the decision matrix
    let normalized_matrix = alternatives.normalize_min_max(&criteria_types)?;

    // Step 2: Apply equal weights to all criteria.
    let equal_weights = normalized_matrix.weight_equal()?;

    // Step 3: Rank alternatives using TOPSIS
    let ranking = normalized_matrix.rank_topsis(&equal_weights)?;

    // Output the ranking (higher values indicate more preferred alternative)
    println!("Ranking: {:.3}", ranking);

    Ok(())
}
```

### Real-World Example: Supplier Selection

This example shows how to select the best supplier based on multiple criteria:

```rust
use mcdm::{CriteriaTypes, McdmError, Normalize, Rank, Weight};
use nalgebra::dmatrix;

fn main() -> Result<(), McdmError> {
    // Decision matrix: Suppliers (rows) x Criteria (columns)
    // Criteria: Cost, Quality, Delivery Time, Reliability, Environmental Impact
    let suppliers = dmatrix![
        78.0, 8.5, 12.0, 9.0, 7.0;  // Supplier A
        65.0, 7.0, 10.0, 8.0, 8.5;  // Supplier B
        82.0, 9.0, 14.0, 7.5, 9.0;  // Supplier C
        70.0, 8.0, 11.0, 9.5, 6.5   // Supplier D
    ];

    // Criteria types: minimize cost and delivery time, maximize others
    let criteria_types = CriteriaTypes::from_slice(&[-1, 1, -1, 1, 1])?;

    // Normalize using Vector normalization
    let normalized = suppliers.normalize_vector(&criteria_types)?;

    // Use entropy weighting to objectively determine criteria importance
    let weights = normalized.weight_entropy()?;
    println!("Criteria weights: {:.3}", weights);

    // Rank suppliers using TOPSIS method
    let rankings = normalized.rank_topsis(&weights)?;

    println!("Supplier rankings (higher is better):");
    println!("Supplier A: {:.4}", rankings[0]);
    println!("Supplier B: {:.4}", rankings[1]);
    println!("Supplier C: {:.4}", rankings[2]);
    println!("Supplier D: {:.4}", rankings[3]);

    // Find the best supplier
    let best_index = rankings
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(index, _)| index)
        .unwrap();

    println!("Best supplier: {}", ['A', 'B', 'C', 'D'][best_index]);

    Ok(())
}
```

## ⚙️ Dependencies

The `mcdm` crate primarily depends on:

- [nalgebra](https://docs.rs/nalgebra/latest/nalgebra/) for matrix operations and data types.
- [error_set](https://docs.rs/error-set/latest/error_set/) for error handling.

## 🌐 Common Use Cases

Multi-criteria decision making is useful in many scenarios, including:

1. **Investment Portfolio Selection:** Evaluating different investment options based on risk, return, liquidity, etc.
2. **Supplier Selection:** Choosing vendors based on cost, quality, delivery time, reliability, environmental impact, etc.
3. **Location Selection:** Determining optimal facility locations based on cost, proximity to markets, labor availability, etc.
4. **Product Design:** Evaluating different design alternatives based on performance, cost, manufacturability, etc.
5. **Environmental Impact Assessment:** Comparing project alternatives based on environmental impact factors.

## 🤝 Contributing

Contributing is what makes the open source community thrive. Any contributions to enhance `mcdm` are welcome and greatly apprecaited!

If you have any suggestions that could make this project better, please feel free to fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

### How to Contribute

1. Fork this repo
2. Create your feature branch (`git checkout -b feature/new-decision-method`)
3. Commit your changes (`git commit -m 'Add new decision method'`)
4. Push to the remote branch (`git push origin feature/new-decision-method`)
5. Open a pull request

### Reporting Bugs

Found a bug? Please open an issue with the "bug" tag and include:

- A clear description of the bug
- Steps to reproduce the bug
- Expected vs. actual behavior
- Your environment (Rust version, OS, mcdm version, etc.)

### Feature Requests

Have an idea for a new method or feature? Open an issue with the "feature" tag and describe what you'd like to see added.

## 📚 Additional Resources

- [Complete API Documentation](https://docs.rs/mcdm/latest/mcdm/)
- [Crate Page on crates.io](https://crates.io/crates/mcdm)
- [MCDM Theory and Background](https://en.wikipedia.org/wiki/Multiple-criteria_decision_analysis)

## 📄 License

Distributed under the MIT license. See [LICENSE-MIT](LICENSE-MIT) for more information.