

use ndarray::{Array2, ArrayView, Ix1};
use num::{Float, Signed};
use ndarray_stats::DeviationExt;

pub enum DistMetric {
    Euclidean,
    Manhattan
}

fn metric_func<T>(metric : DistMetric) -> fn(
    row1 : &ArrayView<T, Ix1>,
    row2 : &ArrayView<T, Ix1>
) -> f64
where
T : Float + Signed + std::ops::AddAssign + std::convert::Into<f64>
{
    match metric {
        DistMetric::Euclidean => euclidean_dist,
        DistMetric::Manhattan => manhattan_dist
    }
}

pub fn euclidean_dist<T>(
    row1 : &ArrayView<T, Ix1>,
    row2 : &ArrayView<T, Ix1>
) -> f64
where
T : Float + Signed + std::ops::AddAssign + std::convert::Into<f64>// + RawData
{
    row1.l2_dist(row2).unwrap()
}

pub fn manhattan_dist<T>(
    row1 : &ArrayView<T, Ix1>,
    row2 : &ArrayView<T, Ix1>
) -> f64
where
T : Float + Signed + std::ops::AddAssign + std::convert::Into<f64>// + RawData
{
    row1.l1_dist(row2).unwrap().into() as f64
}

pub fn calc_dist_matrix<T>(
    arr1: &Array2<T>,
    arr2: &Array2<T>,
    dist_method : fn(
        row1 : &ArrayView<T, Ix1>,
        row2 : &ArrayView<T, Ix1>
    ) -> f64,
) -> Array2<f64>
where
    T : Float + Signed + std::ops::AddAssign
{
    let mut dists : Array2<f64> = Array2::zeros(( arr1.dim().0, arr2.dim().0 ));

    for i in 0..arr1.dim().0 {
        for j in 0..arr2.dim().0 {
            dists.row_mut(i)[j] = dist_method(&arr1.row(i), &arr2.row(j));
        }
    }

    dists
}

/*  frechet calculates the frechet distance between two curves */
pub fn frechet(arr1: &Array2<f64>, arr2: &Array2<f64>, metric : DistMetric) -> f64 {
    let dist_func = metric_func(metric);

    let dist_matrix = calc_dist_matrix(arr1, arr2, dist_func);
    
    frechet_walk(&dist_matrix)
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

pub fn dtw(arr1: &Array2<f64>, arr2: &Array2<f64>, metric : DistMetric) -> f64 {
    let dist_func = metric_func(metric);
    
    let dist_matrix = calc_dist_matrix(arr1, arr2, dist_func);

    dtw_walk(&dist_matrix)
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

#[cfg(test)]
mod tests {
    use super::*;

    use approx::relative_eq;
    use ndarray::array;

    #[test]
    fn test_frechet() {
        let fr = frechet(
            &array![[1.0], [1.0], [3.0]],
            &array![[2.0], [4.0]],
            DistMetric::Euclidean
        );

        assert_eq!(fr, 1.0);

        let fr1 = frechet(
            &array![[1.0], [3.0], [4.0]],
            &array![[1.0], [7.3]],
            DistMetric::Euclidean
        );

        assert_eq!(fr1, 3.3);
    }

    #[test]
    fn test_calc_dist_matrix() {
        let arr1 = array![[0.1, 0.2], [0.3, 0.4]];
        let arr2 = array![[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]];

        let pdist = calc_dist_matrix(&arr1, &arr2, euclidean_dist);

        relative_eq!(pdist.row(0)[0], 0.56568542);
        relative_eq!(pdist.row(0)[1], 0.84852814);
        relative_eq!(pdist.row(0)[2], 1.13137085);
        relative_eq!(pdist.row(1)[0], 0.28284271);
        relative_eq!(pdist.row(1)[1], 0.56568542);
        relative_eq!(pdist.row(1)[2], 0.84852814);
    }
}
