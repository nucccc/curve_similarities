use curve_similarities::{frechet, DistMetric};

use ndarray::array;

#[test]
fn test_frechet_euclidean() {
    let fr = frechet(
        &array![[1.0], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    );

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Euclidean
    );

    assert_eq!(fr1, 3.3);

    let fr2 = frechet(
        &array![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Euclidean
    );

    assert_eq!(fr2, 3.1622776601683795);
}

#[test]
fn test_frechet_euclidean_f32() {
    let fr = frechet(
        &array![[1.0_f32], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    );

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &array![[1.0_f32], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Euclidean
    );

    assert_eq!(fr1, 3.3000001965147017);

    let fr2 = frechet(
        &array![[1.0_f32, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Euclidean
    );

    assert_eq!(fr2, 3.1622776601683795);
}

#[test]
fn test_frechet_manhattan() {
    let fr = frechet(
        &array![[1.0], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Manhattan
    );

    assert_eq!(fr, 1.0);

    let fr1 = frechet(
        &array![[1.0], [3.0], [4.0]],
        &array![[1.0], [7.3]],
        DistMetric::Manhattan
    );

    assert_eq!(fr1, 3.3);

    let fr2 = frechet(
        &array![[1.0, 2.0], [1.0, 4.0], [3.0, 1.0]],
        &array![[2.0, 5.0], [4.0, 2.0]],
        DistMetric::Manhattan
    );

    assert_eq!(fr2, 4.0);
}