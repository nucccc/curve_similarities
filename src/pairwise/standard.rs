/* this file contains code to compute pairwise distances using standard types */

use num::{Float, Signed};

use std::ops::{AddAssign, Sub};

use crate::pairwise::DistMetric;

pub fn euclidean_dist<PD>(
    row1 : &PD,
    row2 : &PD,
) -> Result<f64, String>
where
PD: PairDiff
{
    row1.euclidean(&row2)
}

fn manhattan_dist<PD>(
    row1 : &PD,
    row2 : &PD,
) -> Result<f64, String>
where
PD: PairDiff
{
    row1.manhattan(&row2)
}

/** smetric_func returns the metric function for pairwise distance computations
 * not involving ndarray, but just standard types
 */
pub fn smetric_func<PD>(metric : DistMetric) -> fn(
    row1 : &PD,
    row2 : &PD,
) -> Result<f64, String>
where
PD: PairDiff
{
    match metric {
        DistMetric::Euclidean => euclidean_dist,
        DistMetric::Manhattan => manhattan_dist
    }
}


/**  PairDiff trait provides pairwise distance for various primitive types */
pub trait PairDiff {
    fn euclidean(&self, other: &Self) -> Result<f64, String>;

    fn manhattan(&self, other: &Self) -> Result<f64, String>;
}

impl PairDiff for f64 {
    fn euclidean(&self, other: &Self) -> Result<f64, String> {
        Ok((self - other).abs())
    }

    fn manhattan(&self, other: &Self) -> Result<f64, String> {
        Ok((self - other).abs())
    }
}

impl PairDiff for f32 {
    fn euclidean(&self, other: &Self) -> Result<f64, String> {
        Ok((self - other).abs() as f64)
    }

    fn manhattan(&self, other: &Self) -> Result<f64, String> {
        Ok((self - other).abs() as f64)
    }    
}

impl<T, const N: usize> PairDiff for [T; N]
where T: Copy + Default + Sub<Output = T> + AddAssign + Signed + Float + std::convert::Into<f64>
{
    fn euclidean(&self, other: &Self) -> Result<f64, String> {
        let mut res: T = T::default();

        for i in 0..self.len() {
            res += (self[i] - other[i]).powi(2);
        }

        Ok(res.sqrt().into())
    }

    fn manhattan(&self, other: &Self) -> Result<f64, String> {
        let mut res: T = T::default();

        for i in 0..self.len() {
            res += (self[i] - other[i]).abs();
        }

        Ok(res.into())
    }    
}

/*  just a simple utility to return an error in case vectors have different lens */
fn validate_vec_lens<T>(v1: &Vec<T>, v2: &Vec<T>) -> Result<(), String> {
    if v1.len() != v2.len() {
        return Err(format!("different distance between vectors when calculating pairwise distance, got lens {} {}", v1.len(), v2.len()))
    }

    Ok(())
}

impl<T> PairDiff for Vec<T>
where T: Copy + Default + Sub<Output = T> + AddAssign + Signed + Float + std::convert::Into<f64>
{
    fn euclidean(&self, other: &Self) -> Result<f64, String> {
        validate_vec_lens(self, other)?;
        
        let mut res: T = T::default();

        for i in 0..self.len() {
            res += (self[i] - other[i]).powi(2);
        }

        Ok(res.sqrt().into())
    }

    fn manhattan(&self, other: &Self) -> Result<f64, String> {
        validate_vec_lens(self, other)?;

        let mut res: T = T::default();

        for i in 0..self.len() {
            res += (self[i] - other[i]).abs();
        }

        Ok(res.into())
    }    
}