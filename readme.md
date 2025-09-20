# curve_similarities

Implementation of calculations for curves similarities as in python package [similarity_measures](https://github.com/cjekel/similarity_measures).

The following distances are currently implemented:
- Dynamic Time Warping, from: Senin, P., 2008. Dynamic time warping algorithm review. Information and Computer Science Department University of Hawaii at Manoa Honolulu, USA, 855, pp.1-23 [PDF](http://seninp.github.io/assets/pubs/senin_dtw_litreview_2008.pdf)
- Frechet distance, from: Thomas Eiter and Heikki Mannila, Computing discrete Frechet distance. Technical report, 1994.
- Curve length measure, from: A Andrade-Campos, R De-Carvalho, and R A F Valente. Novel criteria for determination of material model parameters. International Journal of Mechanical Sciences, 54(1):294-305, 2012. ISSN 0020-7403. [DOI](https://doi.org/10.1016/j.ijmecsci.2011.11.010) [URL](http://www.sciencedirect.com/science/article/pii/S0020740311002451)
- Area between two curves, from: Jekel, C. F., Venter, G., Venter, M. P., Stander, N., & Haftka, R. T. (2018). Similarity measures for identifying material parameters from hysteresis loops using inverse analysis. International Journal of Material Forming. [DOI](https://doi.org/10.1007/s12289-018-1421-8)

The library requires in input a bidimensional array of the [ndarray](https://github.com/rust-ndarray/ndarray) rust library.

## Examples

```rust
use curve_similarities::{frechet, DistMetric};
use ndarray::array;


fn main() {
    let fr = frechet(
        &array![[1.0], [1.0], [3.0]],
        &array![[2.0], [4.0]],
        DistMetric::Euclidean
    ).unwrap();

    println!("Frechet distance between curves is {}", fr);
}

```