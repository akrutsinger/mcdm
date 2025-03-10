//! Correlation methods for the `DMatrix` type from the `nalgebra` crate.

#[allow(unused_imports)]
use crate::Float;
use nalgebra::DMatrix;

pub trait Correlate {
    /// Calculate the distance correlation matrix
    ///
    /// The distance correlation matrix is calculated as:
    ///
    /// $$ dCor(c_j, c_{j^\prime}) = \frac{dCov(c_j, c_{j^\prime})}{\sqrt{dVar(c_j)dVar(c_{j^\prime})}} $$
    ///
    /// where $dCov(c_j, c_{j^\prime})$ is the distance covariance between column $c_j$ and
    /// $c_{j^\prime}$, $dVar(c_j) = dCov(c_j, c_j)$ is the distance variance of column $c_j$, and
    /// $dVar(c_{j^\prime}) = dCov(c_{j^\prime}, c_{j^\prime})$ is the distance variance of
    /// $c_{j^\prime}$.
    ///
    /// # Returns
    ///
    /// A 2-dimensional distance correlation matrix of type `DMatrix<f64>`.
    fn distance_correlation(&self) -> DMatrix<f64>;

    /// Calculate the Pearson correlation matrix
    ///
    /// The Pearson correlation matrix is calculated as:
    ///
    /// $$ r_{jk} = \frac{\sum_{i=1}^m (x_{ij} - x_j)(x_{ik} - x_k)}{\sqrt{\sum_{i=1}^m (x_{ij} - x_j)^2}\sqrt{\sum_{i=1}^m (x_{jk} - x_k)^2}} $$
    ///
    /// where $x_{ij}$ is the $i$th element of the row and $j$th elements of the column with $m$
    /// rows and $n$ columns.
    ///
    /// # Returns
    ///
    /// A 2-dimensional Pearson correlation matrix of type `DMatrix<f64>`.
    fn pearson_correlation(&self) -> DMatrix<f64>;
}

impl Correlate for DMatrix<f64> {
    fn distance_correlation(&self) -> DMatrix<f64> {
        let n = self.ncols();
        let mut corr_matrix = DMatrix::zeros(n, n);

        for i in 0..n {
            for j in 0..n {
                let xi = DMatrix::from_column_slice(self.nrows(), 1, self.column(i).as_slice());
                let xj = DMatrix::from_column_slice(self.nrows(), 1, self.column(j).as_slice());

                let dvar_x = distance_covariance(&xi, &xi);
                let dvar_y = distance_covariance(&xj, &xj);

                if dvar_x > 0.0 && dvar_y > 0.0 {
                    corr_matrix[(i, j)] = distance_covariance(&xi, &xj) / (dvar_x * dvar_y).sqrt();
                } else {
                    corr_matrix[(i, j)] = 0.0; // Handle cases where variance is zero
                }
            }
        }

        corr_matrix
    }

    fn pearson_correlation(&self) -> DMatrix<f64> {
        let column_means = self.row_mean();
        let column_stds = self.row_variance().map(|x| x.sqrt());

        let mut normalized_matrix = self.clone();
        for j in 0..self.ncols() {
            let mut col = normalized_matrix.column_mut(j);
            for x in col.iter_mut() {
                *x = (*x - column_means[j]) / column_stds[j];
            }
        }

        normalized_matrix.transpose() * &normalized_matrix / self.nrows() as f64
    }
}

/// Compute the pairwise Euclidean distance matrix for a dataset (column-wise)
fn pairwise_distance_matrix(x: &DMatrix<f64>) -> DMatrix<f64> {
    let m = x.nrows();
    let mut dist_matrix = DMatrix::zeros(m, m);

    for i in 0..m {
        for j in 0..m {
            let diff = x.row(i) - x.row(j);
            dist_matrix[(i, j)] = diff.norm();
        }
    }

    dist_matrix
}

fn double_center_matrix(x: &DMatrix<f64>) -> DMatrix<f64> {
    let m = x.nrows();
    let row_means = x.row_sum() / m as f64;
    let col_means = x.column_sum() / m as f64;
    let overall_mean = x.sum() / (m * m) as f64;

    let mut centered_matrix = x.clone();
    for i in 0..m {
        for j in 0..m {
            centered_matrix[(i, j)] -= row_means[i] + col_means[j] - overall_mean;
        }
    }

    centered_matrix
}
fn distance_covariance(x: &DMatrix<f64>, y: &DMatrix<f64>) -> f64 {
    let ax = double_center_matrix(&pairwise_distance_matrix(x));
    let by = double_center_matrix(&pairwise_distance_matrix(y));

    let m = x.nrows() as f64;
    let dcov2 = (ax.component_mul(&by)).sum() / (m * m);
    dcov2.sqrt()
}
