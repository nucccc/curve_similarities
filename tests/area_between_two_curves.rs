use curve_similarities::area_between_two_curves;

use ndarray::array;

#[test]
fn test_area_between_two_curves() {
    let arr1 = array![[0.1, 0.2], [0.3, 0.4]];
    let arr2 = array![[0.1, 0.6], [0.2, 0.8], [0.3, 1.0]];

    let res = area_between_two_curves(&arr1, &arr2).unwrap();

    assert_eq!(res, 0.09999999999999996);
}

#[test]
fn test_area_between_two_curves2() {
    let arr1 = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let arr2 = array![[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]];

    let res = area_between_two_curves(&arr1, &arr2).unwrap();

    assert_eq!(res, 1.0);
}

#[test]
fn test_area_between_two_curves3() {
    let arr1 = array![[0.1, 0.3], [0.12, 2.0], [0.18, 1.8], [0.2, 0.4]];
    let arr2 = array![[0.1, 0.6], [0.2, 0.7], [0.3, 1.0]];

    let res = area_between_two_curves(&arr1, &arr2).unwrap();

    assert_eq!(res, 0.1909999999999999);
}

#[test]
fn test_area_between_two_curves4() {
    let arr1 = array![
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.3],
        [0.7, 0.2],
        [0.9, 0.2],
        [1.1, 0.2],
    ];
    let arr2 = array![
        [0.5, 0.6],
        [0.6, 1.0],
        [0.7, 0.8],
        [0.8, 0.8],
        [0.9, 1.0],
    ];

    let res = area_between_two_curves(&arr1, &arr2).unwrap();

    assert_eq!(res, 0.3900000000000001);
}

#[test]
fn test_area_between_two_curves_dims_wrong() {
    let arr1 = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    let arr2 = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];

    let res = area_between_two_curves(&arr1, &arr2);
    assert!(res.is_err());
    let res = area_between_two_curves(&arr2, &arr1);
    assert!(res.is_err());

    let arr1 = array![[0.0], [0.0], [0.0], [0.0]];
    let arr2 = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];

    let res = area_between_two_curves(&arr1, &arr2);
    assert!(res.is_err());
    let res = area_between_two_curves(&arr2, &arr1);
    assert!(res.is_err());
}