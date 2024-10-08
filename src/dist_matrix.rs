use ndarray::{Array2, ArrayView, Ix1};
use num::{Float, Signed};
use ndarray_stats::DeviationExt;

pub enum DistMetric {
    Euclidean,
    Manhattan
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

fn manhattan_dist<T>(
    row1 : &ArrayView<T, Ix1>,
    row2 : &ArrayView<T, Ix1>
) -> f64
where
T : Float + Signed + std::ops::AddAssign + std::convert::Into<f64>// + RawData
{
    row1.l1_dist(row2).unwrap().into() as f64
}

pub fn metric_func<T>(metric : DistMetric) -> fn(
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