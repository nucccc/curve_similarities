pub mod metric;
pub mod nstats;
pub mod standard;

pub use metric::DistMetric;
pub use nstats::{euclidean_dist, metric_func};
pub use standard::{PairDiff, smetric_func};