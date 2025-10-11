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
mod pairwise;

pub use pairwise::DistMetric;

pub use area_between_curves::area_between_two_curves;
pub use curve_len::curve_len_measure;
pub use dtw::dtw;
pub use frechet::frechet;