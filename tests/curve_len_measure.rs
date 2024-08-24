use curve_similarities::curve_len_measure;

use ndarray::array;

#[test]
fn test_curve_len_measure() {
    let arr1 = array![[0.1, 0.2], [0.3, 0.4]];
    let arr2 = array![[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]];

    let res = curve_len_measure(&arr1, &arr2);

    assert_eq!(res, 2.248026610499685);
}