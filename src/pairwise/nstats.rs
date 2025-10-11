use ndarray::{ArrayView, Ix1};
use ndarray_stats::DeviationExt;
use num::{Float, Signed};

use std::ops::AddAssign;

use crate::pairwise::DistMetric;

pub fn euclidean_dist<T>(
    row1 : &ArrayView<T, Ix1>,
    row2 : &ArrayView<T, Ix1>
) -> f64
where
T : Float + Signed + AddAssign + std::convert::Into<f64>// + RawData
{
    row1.l2_dist(row2).unwrap()
}

fn manhattan_dist<T>(
    row1 : &ArrayView<T, Ix1>,
    row2 : &ArrayView<T, Ix1>
) -> f64
where
T : Float + Signed + AddAssign + std::convert::Into<f64>// + RawData
{
    row1.l1_dist(row2).unwrap().into()
}


pub fn metric_func<T>(metric : DistMetric) -> fn(
    row1 : &ArrayView<T, Ix1>,
    row2 : &ArrayView<T, Ix1>
) -> f64
where
T : Float + Signed + AddAssign + std::convert::Into<f64>
{
    match metric {
        DistMetric::Euclidean => euclidean_dist,
        DistMetric::Manhattan => manhattan_dist
    }
}