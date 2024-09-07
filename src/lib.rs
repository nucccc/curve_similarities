/*use std::fmt::Debug;
use ndarray::{Array1, Array2, ArrayView, Ix1, ScalarOperand};
use num::{Float, Signed, FromPrimitive, One};
use ndarray_stats::{DeviationExt, QuantileExt};
use ndarray_interp::interp1d::{Interp1D, Linear};*/

mod errors;

mod dist_matrix;

mod area_between_curves;
mod curve_len;
mod dtw;
mod frechet;

pub use dist_matrix::DistMetric;

pub use area_between_curves::is_simple_quad;
pub use curve_len::curve_len_measure;
pub use dtw::dtw;
pub use frechet::frechet;