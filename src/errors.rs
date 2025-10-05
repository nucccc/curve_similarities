use ndarray::{Array2};

pub fn error_dims_str(dim1 : usize, dim2 : usize) -> String {
    format!("Different number of dimensions, array 1 got {dim1} columns, array 2 got {dim2} columns")
}

pub fn validate_two_dim_array<T>(arr: &Array2<T>,) -> Result<(), String> {
    if arr.shape()[1] == 2 {Ok(())} else {Err(
        format!("Got array of shape ({}, {}), expected (:, 2)", arr.shape()[0], arr.shape()[1]).to_string()
    )}
}