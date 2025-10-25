use ndarray::Array2;

use crate::dist_matrix::{DistMatCalc, dist_mat_calc};
use crate::pairwise::DistMetric;

/// Calculates the Dynamic Time Warping
/// 
/// Expects in input:
/// - two arrays/vectors whose elements pairwise distance can be calculated (in
/// this they shall satisfy the `DistMatCalc`)
/// - a `DistMetric` specifying which type of pairwise distance is going to be
/// used
/// 
/// Returns an error in case the row sizes of the two arrays differ
/// 
/// ## Types that implement the `DistMatCalc` trait
/// 
/// - `ndarray::Array1<T>` with `T` being a float
/// - `ndarray::Array2<T>` with `T` being a float
/// - `Vec<T>` with `T` being a float
/// - `Vec<Vec<T>>` with `T` being a float
/// - `Vec<[T; N]>` with `T` being a float and `N` the array size
/// 
/// ## Example
/// ```
/// use curve_similarities::{dtw, DistMetric};
/// use ndarray::array;
/// 
/// 
/// let val = dtw(
///     &array![1.0, 1.0, 3.0],
///     &array![2.0, 4.0],
///     DistMetric::Euclidean
/// ).unwrap();
/// 
/// let val = dtw(
///     &array![[1.0, 2.0], [1.0, 3.0], [3.0, 3.0]],
///     &array![[2.0, 4.0], [4.0, 4.0]],
///     DistMetric::Euclidean
/// ).unwrap();
/// 
/// let val = dtw(
///     &vec![1.0, 1.0, 3.0],
///     &vec![2.0, 4.0],
///     DistMetric::Euclidean
/// ).unwrap();
/// 
/// let val = dtw(
///     &vec![[1.0, 2.0], [1.0, 3.0], [3.0, 3.0]],
///     &vec![[2.0, 4.0], [4.0, 4.0]],
///     DistMetric::Euclidean
/// ).unwrap();
/// 
/// let val = dtw(
///     &vec![vec![1.0, 2.0], vec![1.0, 3.0], vec![3.0, 3.0]],
///     &vec![vec![2.0, 4.0], vec![4.0, 4.0]],
///     DistMetric::Euclidean
/// ).unwrap();
/// ```
pub fn dtw<DMC>(arr1: DMC, arr2: DMC, metric : DistMetric) -> Result<f64, String>
where
    DMC: DistMatCalc
{    
    let dist_matrix = dist_mat_calc(arr1, arr2, metric)?;

    Ok(dtw_walk(&dist_matrix))
}

fn dtw_walk(dist_matrix: &Array2<f64>) -> f64 {
    let n_rows = dist_matrix.dim().0;
    let n_cols = dist_matrix.dim().1;

    let mut ca : Array2<f64> = Array2::zeros( dist_matrix.dim() );

    ca.row_mut(0)[0] = dist_matrix.row(0)[0];

    for i in 1..n_rows {
        ca.row_mut(i)[0] = ca.row(i-1)[0] + dist_matrix.row(i)[0];
    }
    for j in 1..n_cols {
        ca.row_mut(0)[j] = ca.row(0)[j-1] + dist_matrix.row(0)[j];
    }

    for i in 1..n_rows {
        for j in 1..n_cols {
            let mmin = f64::min(ca.row(i-1)[j], ca.row(i)[j-1]);
            let mmmin = f64::min(mmin, ca.row(i-1)[j-1]);
            ca.row_mut(i)[j] = mmmin + dist_matrix.row(i)[j];
        }
    }

    ca.row(n_rows - 1)[n_cols - 1]
}