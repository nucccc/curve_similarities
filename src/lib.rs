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

pub use dist_matrix::DistMatCalc;
pub use pairwise::PairDiff;