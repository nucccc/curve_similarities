use ndarray::{Array1, Array2, ArrayView, Ix1};
use num::{Float, Signed};
use ndarray_stats::DeviationExt;


use crate::errors::error_dims_str;

/** DistMetric represents the possible pairwise distance metrics for elements
to be used when calculating the Frechet distance and Dynamic Time Warping */
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
    row1.l1_dist(row2).unwrap().into()
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

// generic dist matrix calculation

pub fn dist_mat_calc<DMC>(
    a1: DMC,
    a2: DMC,
    metric: DistMetric
) -> Result<Array2<f64>, String>
where DMC: DistMatCalc
{
    a1.dist_mat(&a2, metric)
}

pub trait DistMatCalc {
    fn dist_mat(&self, other: &Self, metric: DistMetric) -> Result<Array2<f64>, String>;
}

impl<T> DistMatCalc for &Array2<T>
where T: Copy + Signed + std::ops::AddAssign + Float + std::convert::Into<f64>
{
    fn dist_mat(&self, other: &Self, metric: DistMetric) -> Result<Array2<f64>, String> {
        // checking for errors
        if self.dim().1 != other.dim().1 {
            return Err(error_dims_str(self.dim().1, other.dim().1));
        }

        if self.is_empty() || other.is_empty() {
            return Err("Input array cannot have 0 length".to_string());
        }
        
        let mut dists : Array2<f64> = Array2::zeros(( self.dim().0, other.dim().0 ));

        let dist_func = metric_func(metric);

        for i in 0..self.dim().0 {
            for j in 0..other.dim().0 {
                println!("{} {}", i, j);
                dists.row_mut(i)[j] = dist_func(&self.row(i), &other.row(j));
            }
        }

        Ok(dists)
    }
}

impl<T> DistMatCalc for &Array1<T>
where T: Copy + Signed + std::ops::AddAssign + Float + std::convert::Into<f64>
{
    fn dist_mat(&self, other: &Self, _metric: DistMetric) -> Result<Array2<f64>, String> {
        if self.is_empty() || other.is_empty() {
            return Err("Input array cannot have 0 length".to_string());
        }
        
        let mut dists : Array2<f64> = Array2::zeros(( self.dim(), other.dim() ));

        for i in 0..self.dim() {
            for j in 0..other.dim() {
                println!("{} {}", i, j);
                dists.row_mut(i)[j] = (self[i] - other[j]).abs().into();
            }
        }

        Ok(dists)
    }
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

        let pdist = dist_mat_calc(&arr1, &arr2, DistMetric::Euclidean).unwrap();

        relative_eq!(pdist.row(0)[0], 0.56568542);
        relative_eq!(pdist.row(0)[1], 0.84852814);
        relative_eq!(pdist.row(0)[2], 1.13137085);
        relative_eq!(pdist.row(1)[0], 0.28284271);
        relative_eq!(pdist.row(1)[1], 0.56568542);
        relative_eq!(pdist.row(1)[2], 0.84852814);
    }

    #[test]
    fn test_dist_matrix_single_dim_array() {
        let arr1 = array![0.1, 0.2];
        let arr2 = array![0.8, 0.9, 1.0];

        let pdist = dist_mat_calc(&arr1, &arr2, DistMetric::Euclidean).unwrap();

        relative_eq!(pdist.row(0)[0], 0.7);
        relative_eq!(pdist.row(0)[1], 0.8);
        relative_eq!(pdist.row(0)[2], 0.9);
        relative_eq!(pdist.row(1)[0], 0.6);
        relative_eq!(pdist.row(1)[1], 0.7);
        relative_eq!(pdist.row(1)[2], 0.8);

        let arr1 = array![0.1, 0.2];
        let arr2 = array![0.8, 0.9, 1.0];

        let pdist = dist_mat_calc(&arr2, &arr1, DistMetric::Euclidean).unwrap();

        relative_eq!(pdist.row(0)[0], 0.7);
        relative_eq!(pdist.row(0)[1], 0.6);
        relative_eq!(pdist.row(1)[0], 0.8);
        relative_eq!(pdist.row(1)[1], 0.7);
        relative_eq!(pdist.row(2)[0], 0.9);
        relative_eq!(pdist.row(2)[1], 0.8);
    }

}