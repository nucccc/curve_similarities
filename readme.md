# curve_similarities

![Version](https://img.shields.io/crates/v/curve_similarities.svg)
![License](https://img.shields.io/crates/l/curve_similarities.svg)

Implementation of calculations for curves similarities as in python package [similarity_measures](https://github.com/cjekel/similarity_measures).

The following distances are currently implemented:
- Dynamic Time Warping, from: Senin, P., 2008. Dynamic time warping algorithm review. Information and Computer Science Department University of Hawaii at Manoa Honolulu, USA, 855, pp.1-23 [PDF](http://seninp.github.io/assets/pubs/senin_dtw_litreview_2008.pdf)
- Frechet distance, from: Thomas Eiter and Heikki Mannila, Computing discrete Frechet distance. Technical report, 1994.
- Curve length measure, from: A Andrade-Campos, R De-Carvalho, and R A F Valente. Novel criteria for determination of material model parameters. International Journal of Mechanical Sciences, 54(1):294-305, 2012. ISSN 0020-7403. [DOI](https://doi.org/10.1016/j.ijmecsci.2011.11.010) [URL](http://www.sciencedirect.com/science/article/pii/S0020740311002451)
- Area between two curves, from: Jekel, C. F., Venter, G., Venter, M. P., Stander, N., & Haftka, R. T. (2018). Similarity measures for identifying material parameters from hysteresis loops using inverse analysis. International Journal of Material Forming. [DOI](https://doi.org/10.1007/s12289-018-1421-8)

The library requires in input a bidimensional array of the [ndarray](https://github.com/rust-ndarray/ndarray) rust library.

## Examples

### DTW and Frechet distances

Frechet and DTW distances require in input two curves with a different number of rows, but with the same length for the rows:

```rust
use curve_similarities::{frechet, DistMetric};
use ndarray::array;


fn main() {
    // calculating frechet distance
    let fr = frechet(
        &array![[1.0], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    ).unwrap();

    println!("Frechet distance between curves is {}", fr);
}

```

As an example one can have curves constituted by samples of 4 values (let's say that we are measuring some whatever with 4 different sensors). In such case every row of the array shall have 4 different values, while the arrays can have a different number of rows: 

```rust
use curve_similarities::{dtw, DistMetric};
use ndarray::array;

fn main() {
    // calculating dynamic time warping
    let dtw_dist = dtw(
        &array![[1.0, 2.0, 3.0, 6.0], [1.0, 4.0, 7.0, 9.0], [3.0, 1.0, -1.0, 2.0]],
        &array![[2.0, 5.0, -7.0, 4.0], [4.0, 2.0, 4.0, 2.0]],
        DistMetric::Euclidean
    ).unwrap();

    println!("Dynamic Time Warping between curves is {}", dtw_dist);
}
```

Since the underlying calculation is based on the pairwise distance between every row, it is possible to specify the pairwise distance metric, which can be either the euclidean or the manhattan distance.

### `curve_len_measure` and `area_between_two_curves`

`curve_len_measure` and `area_between_two_curves` require each one two bidimensional arrays to be provided in input. Each bidimensional array can have the first dimension to be of whatever size, but the second dimension should be of size 2. So every row shall have two elements, the first the value on the x axis and the second being the one on the y axis:

```rust
use curve_similarities::curve_len_measure;
use ndarray::array;


fn main() {
    let arr1 = array![[0.1, 0.2], [0.3, 0.4]];
    let arr2 = array![[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]];

    let res = curve_len_measure(&arr1, &arr2).unwrap();
}

```