use ndarray::{array, s, stack, Array1, Array2, Axis};
use ndarray_stats::QuantileExt;
use num::{Float, FromPrimitive, Signed, Zero};
use ndarray_interp::interp1d::{Interp1D, Linear};

use crate::dist_matrix::euclidean_dist;
use crate::errors::validate_two_dim_array;

/** Calculates the  distance between two curves according to: Jekel, C. F.,
Venter, G., Venter, M. P., Stander, N., & Haftka, R. T. (2018) "Similarity
measures for identifying material parameters from hysteresis loops using
inverse analysis"

Expects two arrays' rows be be of length 2. Having two elements with the first
element being the value on the x axis, while the second element will represent
the value on the y axis of the curve

Returns an error in case any of the two input arrays has rows of length
different than 2 */
pub fn area_between_two_curves<T>(
    arr1: &Array2<T>,
    arr2: &Array2<T>
) -> Result<T, String>
where
T : Float + Signed + std::ops::AddAssign + std::convert::From<i32> + std::convert::Into<f64> + std::fmt::Debug + std::marker::Send + FromPrimitive + 'static
{
    validate_two_dim_array(arr1)?;
    validate_two_dim_array(arr2)?;

    let short = if arr1.shape()[0] < arr2.shape()[0] {arr1} else {arr2};
    let long = if arr1.shape()[0] < arr2.shape()[0] {arr2} else {arr1};

    let longed = enlarge(short, long.shape()[0]);

    let mut area: T = Zero::zero();

    for i in 1..longed.shape()[0] {
        let mut tx: Array1<T> = array![
            long.row(i-1)[0],
            long.row(i)[0],
            longed.row(i)[0],
            longed.row(i-1)[0],
        ];
        let mut ty: Array1<T> = array![
            long.row(i-1)[1],
            long.row(i)[1],
            longed.row(i)[1],
            longed.row(i-1)[1],
        ];

        let mq = make_quad(&mut tx, &mut ty);

        area += mq;
    }

    Ok(area)
}

fn cross_2d<T>(v0_0 : T, v0_1 : T, v1_0 : T, v1_1 : T) -> T
where
T : Float
{
    (v0_0 * v1_1) - (v0_1 * v1_0)
}

#[warn(clippy::too_many_arguments)]
fn is_simple_quad<T>(
    v0_0 : T, v0_1 : T,
    v1_0 : T, v1_1 : T,
    v2_0 : T, v2_1 : T,
    v3_0 : T, v3_1 : T
) -> bool
where
T : Float
{
    let mut ts : [T; 4] = [T::zero(); 4];
    ts[0] = cross_2d(v0_0, v0_1, v1_0, v1_1);
    ts[1] = cross_2d(v1_0, v1_1, v2_0, v2_1);
    ts[2] = cross_2d(v2_0, v2_1, v3_0, v3_1);
    ts[3] = cross_2d(v3_0, v3_1, v0_0, v0_1);

    let mut pos : u8 = 0;
    let mut neg : u8 = 0;
    let mut zer : u8 = 0;

    for elem in ts.iter() {
        if *elem > T::zero() {
            pos += 1;
        } else if *elem < T::zero() {
            neg += 1;
        } else {
            zer += 1;
        }
    }

    let tf = if pos < neg {neg + zer} else {pos + zer};

    tf > 2    
}


fn roll_one<T>(input: &Array1<T>) -> Array1<T>
where
T : Float
{
    let mut res: Array1<T> = Array1::zeros(input.len());

    res[0] = input[input.len()-1];

    for i in 1..input.len() {
        res[i] = input[i-1];
    }

    res
}

fn poly_area<T>(
    x: &Array1<T>,
    y: &Array1<T>,
) -> T
where 
T : Float + Signed + FromPrimitive + 'static
{
    let yr: Array1<T> = roll_one(y);
    let xr: Array1<T> = roll_one(x);
    let half = T::from(0.5).unwrap();

    let num = (x.dot(&yr) - y.dot(&xr)).abs();

    half * num
}


pub fn make_quad<T>(
    x: &mut Array1<T>,
    y: &mut Array1<T>,
) -> T
where 
T : Float + Signed + FromPrimitive + 'static
{
    let mut c: T;

    if ! is_simple_quad(
        x[1]-x[0],
        y[1]-y[0],
        x[2]-x[1],
        y[2]-y[1],
        x[3]-x[2],
        y[3]-y[2],
        x[0]-x[3],
        y[0]-y[3]
    ) {
        c = x[0];
        x[0] = x[1];
        x[1] = c;
        c = y[0];
        y[0] = y[1];
        y[1] = c;

        if ! is_simple_quad(
            x[1]-x[0],
            y[1]-y[0],
            x[2]-x[1],
            y[2]-y[1],
            x[3]-x[2],
            y[3]-y[2],
            x[0]-x[3],
            y[0]-y[3]
        ) {
            c = x[2];
            x[2] = x[0];
            x[0] = x[1];
            x[1] = c;

            c = y[2];
            y[2] = y[0];
            y[0] = y[1];
            y[1] = c;
        }
    }
    
    poly_area(x, y)
}

pub fn arc_len<T>(arr : &Array2<T>) -> Array1<f64>
where
    T : Float + Signed + std::ops::AddAssign + std::convert::Into<f64>// + std::fmt::Debug
{
    let fs = arr.slice(s![..-1, ..]);
    let ls = arr.slice(s![1.., ..]);

    let mut res: Array1<f64> = Array1::zeros(ls.dim().0 );

    for i in 0..ls.dim().0 {
        res[i] = euclidean_dist(&ls.row(i), &fs.row(i))
    }

    res
}

fn enlarge<T>(arr : &Array2<T>, desired_size : usize) -> Array2<T>
where
    T : Float + Signed + std::ops::AddAssign + std::convert::From<i32> + std::convert::Into<f64> + std::fmt::Debug + std::marker::Send
{
    let mut dist_arr = arc_len(arr);

    let dist_arr_original = dist_arr.clone();

    let mut divs_counter = vec![1; dist_arr.dim()];

    for _ in arr.dim().0..(desired_size) {
        let to_div = dist_arr.argmax().unwrap();
        divs_counter[to_div] += 1;
        dist_arr[to_div] = dist_arr_original[to_div] / (divs_counter[to_div] as f64);
    }

    let x = arr.slice(s![.., 0]);
    let y = arr.slice(s![.., 1]);

    let mut xi : usize = 0;
    let mut new_x : Array1<T> = Array1::zeros(desired_size );

    new_x[xi] = x[xi];
    xi += 1;

    for i in 0..divs_counter.len() {
        if divs_counter[i] > 1 {
            let portion = (x[i+1] - x[i]) / (T::try_from(divs_counter[i]).unwrap());
            for mult in 1..divs_counter[i] {
                new_x[xi] = x[i] + (portion * T::try_from(mult).unwrap());
                xi += 1;
            }
        }

        new_x[xi] = x[i+1];
        xi += 1;
    }

    let interpolator = Interp1D::builder(y)
        .x(x)
        .strategy(Linear::new())
        .build()
        .unwrap();

    let new_y = interpolator.interp_array(&new_x).unwrap();

    // TODO: maybe there is a faster way to return this array
    stack(Axis(1), &[new_x.view(), new_y.view()]).unwrap().into_shape_clone((desired_size, 2)).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;

    #[test]
    fn test_arc_len() {
        let c = arc_len(&array![
            [0.0, 0.2],
            [0.1, 0.3],
            [0.2, 0.2],
            [0.3, 0.4],
            [0.4, 0.5],
            [0.5, 0.2]
        ]);

        assert!(c == array![
            0.1414213562373095,
            0.1414213562373095,
            0.22360679774997896,
            0.14142135623730953,
            0.31622776601683794
        ]);
    }

    #[test]
    fn test_enlarge() {
        let c = enlarge(&array![
            [0.0, 0.2],
            [0.1, 0.3],
            [0.2, 0.2],
            [0.3, 0.4],
            [0.4, 0.5],
            [0.5, 0.2]
        ],
        7
    );

        assert!(c == array![
            [0.0, 0.2],
            [0.1, 0.3],
            [0.2, 0.2],
            [0.3, 0.4],
            [0.4, 0.5],
            [0.45, 0.35],
            [0.5, 0.2],
        ]);
    }

    #[test]
    fn test_is_simple_quad0() {
        let x = [0.0, 1.0, 1.0, 0.0];
        let y = [0.0, 1.0, 0.0, 1.0];

        let ab = [x[1]-x[0], y[1]-y[0]];
        let bc = [x[2]-x[1], y[2]-y[1]];
        let cd = [x[3]-x[2], y[3]-y[2]];
        let da = [x[0]-x[3], y[0]-y[3]];

        let quad = is_simple_quad(
            ab[0], ab[1],
            bc[0], bc[1],
            cd[0], cd[1],
            da[0], da[1],
        );
        
        assert!(!quad);
    }

    #[test]
    fn test_is_simple_quad1() {
        let x = [0.0, 0.0, 1.0, 1.0];
        let y = [0.0, 1.0, 1.0, 0.0];

        let ab = [x[1]-x[0], y[1]-y[0]];
        let bc = [x[2]-x[1], y[2]-y[1]];
        let cd = [x[3]-x[2], y[3]-y[2]];
        let da = [x[0]-x[3], y[0]-y[3]];

        let quad = is_simple_quad(
            ab[0], ab[1],
            bc[0], bc[1],
            cd[0], cd[1],
            da[0], da[1],
        );
        
        assert!(quad);
    }

    #[test]
    fn test_is_simple_quad2() {
        let x = [0.0, 1.0, 1.0, 0.0];
        let y = [0.0, 0.0, 1.0, 0.0];

        let ab = [x[1]-x[0], y[1]-y[0]];
        let bc = [x[2]-x[1], y[2]-y[1]];
        let cd = [x[3]-x[2], y[3]-y[2]];
        let da = [x[0]-x[3], y[0]-y[3]];

        let quad = is_simple_quad(
            ab[0], ab[1],
            bc[0], bc[1],
            cd[0], cd[1],
            da[0], da[1],
        );
        
        assert!(quad);
    }

    #[test]
    fn test_is_simple_quad3() {
        let x = [0.0, 1.0, 1.0, 0.0];
        let y = [0.0, 1.0, 0.0, 0.0];

        let ab = [x[1]-x[0], y[1]-y[0]];
        let bc = [x[2]-x[1], y[2]-y[1]];
        let cd = [x[3]-x[2], y[3]-y[2]];
        let da = [x[0]-x[3], y[0]-y[3]];

        let quad = is_simple_quad(
            ab[0], ab[1],
            bc[0], bc[1],
            cd[0], cd[1],
            da[0], da[1],
        );
        
        assert!(quad);
    }

    #[test]
    fn test_make_quad() {
        let mut x = array![0.1, 0.2, 0.12, 0.1];
        let mut y = array![0.6, 0.7, 2.0, 0.3];

        let mq = make_quad(&mut x, &mut y);
        
        assert_eq!(mq, 0.08399999999999999);
    }
}
