use curve_similarities::{dtw, DistMetric};

use ndarray::array;

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
fn test_dtw_wrong_dims() {
    let dtw_err = dtw(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0, 2.0], [7.3, 3.7]],
        DistMetric::Euclidean
    );

    assert!(dtw_err.is_err());
}