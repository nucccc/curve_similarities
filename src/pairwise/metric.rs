/** DistMetric represents the possible pairwise distance metrics for elements
to be used when calculating the Frechet distance and Dynamic Time Warping */
pub enum DistMetric {
    Euclidean,
    Manhattan
}