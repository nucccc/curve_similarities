use std::fmt::Debug;
use ndarray::{Array1, Array2, ArrayView, Ix1, ScalarOperand};
use num::{Float, Signed, FromPrimitive, One};
use ndarray_stats::QuantileExt;
use ndarray_interp::interp1d::{Interp1D, Linear};

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