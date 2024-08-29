use std::fmt::Debug;
use ndarray::{Array1, Array2, ArrayView, Ix1, ScalarOperand};
use num::{Float, Signed, FromPrimitive, One};
use ndarray_stats::{DeviationExt, QuantileExt};
use ndarray_interp::interp1d::{Interp1D, Linear};

fn error_dims_str(dim1 : usize, dim2 : usize) -> String {
    format!("Different number of dimensions, array 1 got {} columns, array 2 got {} columns", dim1, dim2)
}

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

pub fn dtw<T>(arr1: &Array2<T>, arr2: &Array2<T>, metric : DistMetric) -> Result<f64, String>
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

fn calc_length<T>(
    x : &ArrayView<T, Ix1>,
    y : &ArrayView<T, Ix1>,
    norm_seg_len : bool
) -> (Array1<T>, Array1<T>)
where
    T : Float + Signed + std::ops::AddAssign + Default
{
    let n = x.len();

    let mut x_max : T = One::one();
    let mut y_max : T = One::one();

    if norm_seg_len {
        x_max = *x.map(|val| val.abs()).max().unwrap();
        y_max = *y.map(|val| val.abs()).max().unwrap();

        if x_max == T::zero() {
            x_max += T::from(1e-15).unwrap();
        }
        if y_max == T::zero() {
            y_max += T::from(1e-15).unwrap();
        }
    }

    let mut le : Array1<T> = Array1::zeros( n );
    let mut l_sum : Array1<T> = Array1::zeros( n );

    for i in 0..(n-1) {
        le [i + 1] = ((((x[i + 1] - x[i]) / x_max)).powi(2) + (((y[i + 1] - y[i]) / y_max)).powi(2)).sqrt();
        l_sum[i + 1] = l_sum[i] + le[i + 1];
    }

    // TODO: remove, and return the tuple as in the original implementation
    (le, l_sum)
}

pub fn curve_len_measure<T>(arr1: &Array2<T>, arr2: &Array2<T>) -> f64
where
    T : Float + Signed + std::ops::AddAssign + Default + FromPrimitive + ScalarOperand + Debug + std::marker::Send + std::ops::Add<T> + std::convert::Into<f64>
{
    let x1 = arr1.column(0);
    let y1 = arr1.column(1);
    let x2 = arr2.column(0);
    let y2 = arr2.column(1);

    let x_mean = x1.mean().unwrap();
    let y_mean = y1.mean().unwrap();

    let (_, l_sum1) = calc_length(&x1, &y1, true);
    let (_, l_sum2) = calc_length(&x2, &y2, true);

    let li1q = l_sum1.clone() * (l_sum2.sum() / l_sum1.sum());

    let x_interpolator = Interp1D::builder(x2.clone())
        .x(l_sum2.clone())
        .strategy(Linear::new())
        .build()
        .unwrap();

    let y_interpolator = Interp1D::builder(y2.clone())
        .x(l_sum2.clone())
        .strategy(Linear::new())
        .build()
        .unwrap();

    let t_interp_max = *l_sum2.max().unwrap();
    let t_interp_min = *l_sum2.min().unwrap();

    let new_li1q = li1q
        .map(|val| T::min(*val, t_interp_max))
        .map(|val| T::max(*val, t_interp_min));
    
    let x_interp = x_interpolator.interp_array(&new_li1q).unwrap();
    let y_interp = y_interpolator.interp_array(&new_li1q).unwrap();

    let x_interp_minus = ((x_interp - x1)
        .map(|val| val.abs()) / x_mean)
        .map(|val| *val + T::one())
        .map(|val| val.ln())
        .map(|val| val.powi(2));
    let y_interp_minus = ((y_interp - y1)
        .map(|val| val.abs()) / y_mean)
        .map(|val| *val + T::one())
        .map(|val| val.ln())
        .map(|val| val.powi(2));

    (x_interp_minus + y_interp_minus).sum().sqrt().into() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::relative_eq;
    use ndarray::array;

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
