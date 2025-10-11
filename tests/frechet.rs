use curve_similarities::{frechet, DistMetric};

use ndarray::{Array2, array};

#[test]
fn test_frechet_euclidean() {
    let fr = frechet(
        &array![[1.0], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr1, 3.3);

    let fr2 = frechet(
        &array![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr2, 3.1622776601683795);

    let fr3 = frechet(
        &array![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0]],
        &array![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr3, 10.677078252031311);

    let fr4 = frechet(
        &array![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0], [12.0, 15.0, 1.0, 1.0], [16.0, 2.0, 1.0, 1.0]],
        &array![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0], [1.0, 2.0, 7.0, -1.0], [1.0, 2.0, 3.0, 4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr4, 16.278820596099706);

    let fr5 = frechet(        
        &array![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0], [1.0, 2.0, 7.0, -1.0], [1.0, 2.0, 3.0, 4.0]],
        &array![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0], [12.0, 15.0, 1.0, 1.0], [16.0, 2.0, 1.0, 1.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr5, 16.278820596099706);
}

#[test]
fn test_frechet_euclidean_single_dim_array() {
    let fr = frechet(
        &array![1.0, 1.0, 3.0],
        &array![2.0, 4.0],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &array![1.0, 3.0, 4.0],
        &array![1.0, 7.3],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr1, 3.3);
}

#[test]
fn test_frechet_euclidean_vec_single() {
    let fr = frechet(
        &vec![1.0, 1.0, 3.0],
        &vec![2.0, 4.0],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &vec![1.0, 3.0, 4.0],
        &vec![1.0, 7.3],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr1, 3.3);
}

#[test]
fn test_frechet_euclidean_vec_arr() {
    let fr = frechet(
        &vec![[1.0], [1.0], [3.0]],
        &vec![[2.0], [4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &vec![[1.0], [3.0], [4.0]],
        &vec![[1.0], [7.3]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr1, 3.3);

    let fr2 = frechet(
        &vec![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &vec![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr2, 3.1622776601683795);

    let fr3 = frechet(
        &vec![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0]],
        &vec![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr3, 10.677078252031311);

    let fr4 = frechet(
        &vec![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0], [12.0, 15.0, 1.0, 1.0], [16.0, 2.0, 1.0, 1.0]],
        &vec![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0], [1.0, 2.0, 7.0, -1.0], [1.0, 2.0, 3.0, 4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr4, 16.278820596099706);

    let fr5 = frechet(        
        &vec![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0], [1.0, 2.0, 7.0, -1.0], [1.0, 2.0, 3.0, 4.0]],
        &vec![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0], [12.0, 15.0, 1.0, 1.0], [16.0, 2.0, 1.0, 1.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr5, 16.278820596099706);
}

#[test]
fn test_frechet_euclidean_vec_vec() {
    let fr = frechet(
        &vec![vec![1.0], vec![1.0], vec![3.0]],
        &vec![vec![2.0], vec![4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &vec![vec![1.0], vec![3.0], vec![4.0]],
        &vec![vec![1.0], vec![7.3]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr1, 3.3);

    let fr2 = frechet(
        &vec![vec![1.0, 2.0], vec![1.0, 4.0], vec![3.0, 1.0]],
        &vec![vec![2.0, 5.0], vec![4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr2, 3.1622776601683795);

    let fr3 = frechet(
        &vec![vec![1.0, 2.0, 3.0, 6.0], vec![1.0, 4.0, 7.0, 9.0], vec![3.0, 1.0, -1.0, 2.0]],
        &vec![vec![2.0, 5.0, -7.0, 4.0], vec![4.0, 2.0, 4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr3, 10.677078252031311);

    let fr4 = frechet(
        &vec![vec![1.0, 2.0, 3.0, 6.0], vec![1.0, 4.0, 7.0, 9.0], vec![3.0, 1.0, -1.0, 2.0], vec![12.0, 15.0, 1.0, 1.0], vec![16.0, 2.0, 1.0, 1.0]],
        &vec![vec![2.0, 5.0, -7.0, 4.0], vec![4.0, 2.0, 4.0, 2.0], vec![1.0, 2.0, 7.0, -1.0], vec![1.0, 2.0, 3.0, 4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr4, 16.278820596099706);

    let fr5 = frechet(        
        &vec![vec![2.0, 5.0, -7.0, 4.0], vec![4.0, 2.0, 4.0, 2.0], vec![1.0, 2.0, 7.0, -1.0], vec![1.0, 2.0, 3.0, 4.0]],
        &vec![vec![1.0, 2.0, 3.0, 6.0], vec![1.0, 4.0, 7.0, 9.0], vec![3.0, 1.0, -1.0, 2.0], vec![12.0, 15.0, 1.0, 1.0], vec![16.0, 2.0, 1.0, 1.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr5, 16.278820596099706);
}

#[test]
fn test_frechet_euclidean_vec_vec_err() {
    // just to ensure that if a subvector has a different len an error is returned
    let res = frechet(
        &vec![vec![1.0, 2.0], vec![1.0, 4.0], vec![3.0, 1.0]],
        &vec![vec![2.0, 5.0], vec![4.0, 2.0, 5.0]],
        DistMetric::Euclidean
    );

    assert!(res.is_err());
}

#[test]
fn test_frechet_euclidean_f32() {
    let fr = frechet(
        &array![[1.0_f32], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &array![[1.0_f32], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr1, 3.3000001965147017);

    let fr2 = frechet(
        &array![[1.0_f32, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(fr2, 3.1622776601683795);
}

#[test]
fn test_frechet_manhattan() {
    let fr = frechet(
        &array![[1.0], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Manhattan
    ).unwrap();

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Manhattan
    ).unwrap();

    assert_eq!(fr1, 3.3);

    let fr2 = frechet(
        &array![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Manhattan
    ).unwrap();

    assert_eq!(fr2, 4.0);
}

#[test]
fn test_frechet_wrong_dims() {
    let fr_err = frechet(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0, 2.0], [7.3, 3.7]],
        DistMetric::Manhattan
    );

    assert!(fr_err.is_err());
}

#[test]
fn test_frechet_empty() {
    let arr1: Array2<f64> = Array2::<f64>::default((0, 2));
    let arr2: Array2<f64> = Array2::<f64>::default((0, 2));

    let frechet_err = frechet(
        &arr1,
        &arr2,
        DistMetric::Euclidean
    );

    assert!(frechet_err.is_err());

    let frechet_err = frechet(
        &array![[1.0], [3.0], [4.0]],
        &arr2,
        DistMetric::Euclidean
    );

    assert!(frechet_err.is_err());

    let frechet_err = frechet(
        &arr1,
        &array![[1.0], [3.0], [4.0]],
        DistMetric::Euclidean
    );

    assert!(frechet_err.is_err());
}