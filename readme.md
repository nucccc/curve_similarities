# curve_similarities

Implementation of calculations for curves similarities as in python package [similarity_measures](https://github.com/cjekel/similarity_measures).

At the moment has implementations for Frechet and DTW distances, allowing for both Euclidean and Manhattan pointwise distances.

The library requires in input a bidimensional array of the [ndarray](https://github.com/rust-ndarray/ndarray) rust library.

## Examples

```
use curve_similarities::{frechet, DistMetric};
use ndarray::array;


fn main() {
    let fr = frechet(
        &array![[1.0], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    );

    println!("Frechet distance between curves is {}", fr);
}

```

## Dependencies

Currently still using `ndarray` version `0.15.0` in order to use `ndarray-stats` for pointwise distance.

```
[dependencies]
approx = "0.3.2"
ndarray = "0.15.0"
ndarray-stats = "0.5.1"
num = "0.4.3"
```