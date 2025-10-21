use curve_similarities::{dtw, DistMetric};

use ndarray::{Array2, array};

#[test]
fn test_dtw_euclidean() {
    let dtw0 = dtw(
        &array![[1.0], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw0, 3.0);

    let dtw1 = dtw(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw1, 5.3);

    let dtw0 = dtw(
        &array![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw0, 5.99070478491457);

    let dtw3 = dtw(
        &array![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0]],
        &array![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw3, 24.299380447914302);

    let dtw4 = dtw(
        &array![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0], [12.0, 15.0, 1.0, 1.0], [16.0, 2.0, 1.0, 1.0]],
        &array![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0], [1.0, 2.0, 7.0, -1.0], [1.0, 2.0, 3.0, 4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw4, 57.89253119304076);

    let dtw5 = dtw(        
        &array![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0], [1.0, 2.0, 7.0, -1.0], [1.0, 2.0, 3.0, 4.0]],
        &array![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0], [12.0, 15.0, 1.0, 1.0], [16.0, 2.0, 1.0, 1.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw5, 57.89253119304076);
}

#[test]
fn test_dtw_euclidean_vec_arr() {
    let dtw0 = dtw(
        &vec![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &vec![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw0, 5.99070478491457);

    let dtw3 = dtw(
        &vec![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0]],
        &vec![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw3, 24.299380447914302);

    let dtw4 = dtw(
        &vec![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0], [12.0, 15.0, 1.0, 1.0], [16.0, 2.0, 1.0, 1.0]],
        &vec![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0], [1.0, 2.0, 7.0, -1.0], [1.0, 2.0, 3.0, 4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw4, 57.89253119304076);

    let dtw5 = dtw(        
        &vec![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0], [1.0, 2.0, 7.0, -1.0], [1.0, 2.0, 3.0, 4.0]],
        &vec![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0], [12.0, 15.0, 1.0, 1.0], [16.0, 2.0, 1.0, 1.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw5, 57.89253119304076);
}

#[test]
fn test_dtw_euclidean_vec_vec() {
    let dtw0 = dtw(
        &vec![vec![1.0, 2.0], vec![1.0, 4.0], vec![3.0, 1.0]],
        &vec![vec![2.0, 5.0], vec![4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw0, 5.99070478491457);

    let dtw3 = dtw(
        &vec![vec![1.0, 2.0, 3.0, 6.0], vec![1.0, 4.0, 7.0, 9.0], vec![3.0, 1.0, -1.0, 2.0]],
        &vec![vec![2.0, 5.0, -7.0, 4.0], vec![4.0, 2.0, 4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw3, 24.299380447914302);

    let dtw4 = dtw(
        &vec![vec![1.0, 2.0, 3.0, 6.0], vec![1.0, 4.0, 7.0, 9.0], vec![3.0, 1.0, -1.0, 2.0], vec![12.0, 15.0, 1.0, 1.0], vec![16.0, 2.0, 1.0, 1.0]],
        &vec![vec![2.0, 5.0, -7.0, 4.0], vec![4.0, 2.0, 4.0, 2.0], vec![1.0, 2.0, 7.0, -1.0], vec![1.0, 2.0, 3.0, 4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw4, 57.89253119304076);

    let dtw5 = dtw(        
        &vec![vec![2.0, 5.0, -7.0, 4.0], vec![4.0, 2.0, 4.0, 2.0], vec![1.0, 2.0, 7.0, -1.0], vec![1.0, 2.0, 3.0, 4.0]],
        &vec![vec![1.0, 2.0, 3.0, 6.0], vec![1.0, 4.0, 7.0, 9.0], vec![3.0, 1.0, -1.0, 2.0], vec![12.0, 15.0, 1.0, 1.0], vec![16.0, 2.0, 1.0, 1.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw5, 57.89253119304076);
}

#[test]
fn test_dtw_euclidean_single_dim_array() {
    let dtw0 = dtw(
        &array![1.0, 1.0, 3.0],
        &array![2.0, 4.0],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw0, 3.0);

    let dtw1 = dtw(
        &array![1.0, 3.0, 4.0],
        &array![1.0, 7.3],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw1, 5.3);
}

#[test]
fn test_dtw_euclidean_single_dim_vec() {
    let dtw0 = dtw(
        &vec![1.0, 1.0, 3.0],
        &vec![2.0, 4.0],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw0, 3.0);

    let dtw1 = dtw(
        &vec![1.0, 3.0, 4.0],
        &vec![1.0, 7.3],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw1, 5.3);
}

#[test]
fn test_dtw_euclidean_single_dim_vec_f32() {
    let dtw0 = dtw(
        &vec![1.0_f32, 1.0, 3.0],
        &vec![2.0_f32, 4.0],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw0, 3.0);

    let dtw1 = dtw(
        &vec![1.0_f32, 3.0, 4.0],
        &vec![1.0_f32, 7.3],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw1, 5.300000190734863);
}

#[test]
fn test_dtw_euclidean_f32() {
    let dtw0 = dtw(
        &array![[1.0_f32], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw0, 3.0);

    let dtw1 = dtw(
        &array![[1.0_f32], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw1, 5.300000196514702);

    let dtw0 = dtw(
        &array![[1.0_f32, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    assert_eq!(dtw0, 5.99070478491457);
}


#[test]
fn test_dtw_manhattan() {
    let dtw0 = dtw(
        &array![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Manhattan
    ).unwrap();

    assert_eq!(dtw0, 8.0);

    let dtw1 = dtw(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Manhattan
    ).unwrap();

    assert_eq!(dtw1, 5.3);
}

#[test]
fn test_dtw_euclidean_vec_vec_err() {
    // just to ensure that if a subvector has a different len an error is returned
    let res = dtw(
        &vec![vec![1.0, 2.0], vec![1.0, 4.0], vec![3.0, 1.0]],
        &vec![vec![2.0, 5.0], vec![4.0, 2.0, 5.0]],
        DistMetric::Euclidean
    );

    assert!(res.is_err());
}

#[test]
fn test_dtw_wrong_dims() {
    let dtw_err = dtw(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0, 2.0], [7.3, 3.7]],
        DistMetric::Euclidean
    );

    assert!(dtw_err.is_err());
}

#[test]
fn test_dtw_empty() {
    let arr1: Array2<f64> = Array2::<f64>::default((0, 2));
    let arr2: Array2<f64> = Array2::<f64>::default((0, 2));

    let dtw_err = dtw(
        &arr1,
        &arr2,
        DistMetric::Euclidean
    );

    assert!(dtw_err.is_err());

    let dtw_err = dtw(
        &array![[1.0], [3.0], [4.0]],
        &arr2,
        DistMetric::Euclidean
    );

    assert!(dtw_err.is_err());

    let dtw_err = dtw(
        &arr1,
        &array![[1.0], [3.0], [4.0]],
        DistMetric::Euclidean
    );

    assert!(dtw_err.is_err());
}