pub fn error_dims_str(dim1 : usize, dim2 : usize) -> String {
    format!("Different number of dimensions, array 1 got {} columns, array 2 got {} columns", dim1, dim2)
}