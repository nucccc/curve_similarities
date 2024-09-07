use ndarray::Array2;
use num::{Float, Signed};

use crate::dist_matrix::{metric_func, calc_dist_matrix, DistMetric};
use crate::errors::error_dims_str;

/*  frechet calculates the frechet distance between two curves */
pub fn frechet<T>(arr1: &Array2<T>, arr2: &Array2<T>, metric : DistMetric) -> Result<f64, String>
where
    T : Float + Signed + std::ops::AddAssign + std::convert::Into<f64>
{
    let dim1 = arr1.dim().1;
    let dim2 = arr2.dim().1;
    if dim1 != dim2 {
        return Err(error_dims_str(arr1.dim().1, arr2.dim().1));
    }

    let dist_func = metric_func(metric);

    let dist_matrix = calc_dist_matrix(arr1, arr2, dist_func);
    
    Ok(frechet_walk(&dist_matrix))
}

fn frechet_walk(dist_matrix: &Array2<f64>) -> f64 {
    let n_rows = dist_matrix.dim().0;
    let n_cols = dist_matrix.dim().1;

    let mut ca : Array2<f64> = - Array2::ones( dist_matrix.dim() );

    ca.row_mut(0)[0] = dist_matrix.row(0)[0];

    for i in 1..n_rows {
        ca.row_mut(i)[0] = f64::max(ca.row(i-1)[0], dist_matrix.row(i)[0]);
    }
    for j in 1..n_cols {
        ca.row_mut(0)[j] = f64::max(ca.row(0)[j-1], dist_matrix.row(0)[j]);
    }

    for i in 1..n_rows {
        for j in 1..n_cols {
            let mmin = f64::min(ca.row(i-1)[j], ca.row(i)[j-1]);
            let mmmin = f64::min(mmin, ca.row(i-1)[j-1]);
            ca.row_mut(i)[j] = f64::max(mmmin, dist_matrix.row(i)[j]);
        }
    }

    ca.row(n_rows - 1)[n_cols - 1]
}