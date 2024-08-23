use curve_similarities::{dtw, DistMetric};

use ndarray::array;

#[test]
fn test_dtw_euclidean() {
    let dtw0 = dtw(
        &array![[1.0], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    );

    assert_eq!(dtw0, 3.0);

    let dtw1 = dtw(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Euclidean
    );

    assert_eq!(dtw1, 5.3);

    let dtw0 = dtw(
        &array![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Euclidean
    );

    assert_eq!(dtw0, 5.99070478491457);
}

#[test]
fn test_dtw_manhattan() {
    let dtw0 = dtw(
        &array![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Manhattan
    );

    assert_eq!(dtw0, 8.0);

    let dtw1 = dtw(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Manhattan
    );

    assert_eq!(dtw1, 5.3);
}