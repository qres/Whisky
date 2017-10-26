
/// Marks if something is symmetric
/// e.g.: Let M^-1 be a preconditioner for A and M^-1: Symmetric
/// this implies that if A is symmetric PA is so too
pub trait Symmetric{}

pub trait PositiveDefinite{}

/// Marks if somehting is linear.
/// This might not be the case for all solvers used as preconditioner if the solver
/// contains contitions which cause the solver to break early
/// e.g. Multigrid where the iteration number (total number of smoothing steps) depends
/// on the residual
pub trait Linear{}
