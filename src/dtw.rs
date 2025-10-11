use ndarray::Array2;

use crate::dist_matrix::{DistMatCalc, DistMetric, dist_mat_calc};

/** Calculates the Dynamic Time Warping

Expects in input two arrays which can have a different number of rows,
but expects their rows to have the same size, since every row from the first
input array will have its distance calculated from the rows of the second
input array

Returns an error in case the row sizes of the two arrays differ*/
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