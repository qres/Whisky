use ndarray::{Array1, Array2};
use marke::{Linear, Symmetric};
use ::{SolverImpl, Preconditioner, Matrix, Vector};

use std::cmp;

///
/// A = M + N
/// x = M^-1 (N x - b)
///
pub trait AddSplitting {
    type Implicit;
    type Explicit;
    fn implicit(&self) -> &Self::Implicit;
    fn explicit(&self) -> &Self::Explicit;
}

pub struct AdditiveSplitMatrix<Mi,Me> {
    pub implicit: Mi,
    pub explicit: Me,
}

impl<Mi,Me> AddSplitting for AdditiveSplitMatrix<Mi,Me> {
    type Implicit = Mi;
    type Explicit = Me;
    fn implicit(&self) -> &Self::Implicit {&self.implicit}
    fn explicit(&self) -> &Self::Explicit {&self.explicit}

}

///
/// A = M + N
/// A x = b
///
/// (M + N) x = b
/// x' = M^-1 (b - N x)
/// we can use this iteration scheme (M_inv N sub (omega: + add))
///
/// x' = M^-1 (b - (A - M) x)
/// x' = M^-1 (r + M x) | r = b - A x
/// x' = x + M^-1 r | r = b - A x
/// or this iteration scheme (M_inv A sub add (omega: + 0))
/// more expensive (A-mul vs N-mul) but we get the residual for free
///
/// x <- (1-omega) x + omega x'
///
///
/// x_0 is always zero
/// max_iters = 1 -> Normal preconditioner with M = implicit => ((1-w) + w * M^-1) b
///                                                      w=0 => M^-1 b
/// max_iters = 2 -> (1-w)^2 + (1-w)w * M^-1 + w * M^-1 - (1-w)w * M^-1 N + w * w * M^-1 N  M^-1
///             w=1 -> (M^-1 + M^-1 N  M^-1) b
/// max:iters = 3
///             w=1 -> (M^-1 + M^-1 N  M^-1 + M^-1 N  M^-1 N  M^-1) b
///             w=1 -> (I + M^-1 N + (M^-1 N)^2) M^-1 b
pub struct AddSplittingIteration<M> {
    pub max_iters: usize,
    pub omega: f64,
    pub A: M
}

impl<M,Mi,Me,V> Preconditioner<V> for AddSplittingIteration<M>
where
    M: AddSplitting<Implicit=Mi,Explicit=Me>,
    Mi: Preconditioner<V, A=[V]>,
    Me: Matrix<V>,
    V: Vector,
    /*
    for the second version
    M: Matrix<V> + AddSplitting<Implicit=Mi,Explicit=Me>,
    Mi: Preconditioner<V>,
    */
{
    type A = [V];
    fn size(&self) -> usize {
        println!("WARNING!!! we should propably stop to pass &A around or this should be a preconditioner");
        1 + 10
    }

    fn apply(&self, x: &mut V, b: &V, vecs: &mut [V]) {
        let (mut vecs, mut rest) = vecs.split_at_mut(2);
        let mut vecs = vecs.iter_mut();
        let mut x_ = vecs.next().expect("vector buffer too small");
        let mut y = vecs.next().expect("vector buffer too small");

        if self.max_iters == 0 {
            x.set_copy(b)
        } else {
            //// x = 0
            x.set_ax(0.0, b);

            for _ in 0..self.max_iters {
                //// y = b - N x
                Me::aAxpby(&mut y, -1.0, &self.A.explicit(), &x, 1.0, &b);
                //// x = M^-1 y
                self.A.implicit().apply(x_, &y, &mut rest);
                //// x_k+1 = (1-w) x_k + w * x'_k
                x.acc_mul_bx(1.0 - self.omega, self.omega, &x_);
            }
        }
    }
}

impl<T,M,Mi,Me,V> SolverImpl<T,V> for AddSplittingIteration<M>
where
    M: AddSplitting<Implicit=Mi,Explicit=Me>,
    Mi: Preconditioner<V,A=[V]>,
    Me: Matrix<V>,
    V: Vector,
    /*
    for the second version
    M: Matrix<V> + AddSplitting<Implicit=Mi,Explicit=Me>,
    Mi: Preconditioner<V>,
    */
{
    type A = [V];

    fn size(&self) -> usize {
        2 + self.A.implicit().size()
    }

    fn solve(&self, _: &T, x: &mut V, b: &V, vecs: &mut [V]) {
        // TODO println!("WARNING!!! we should propably stop to pass &A around or this should be a preconditioner");

        let (mut vecs, mut rest) = vecs.split_at_mut(2);
        let mut vecs = vecs.iter_mut();
        let mut x_ = vecs.next().expect("vector buffer too small");
        let mut y = vecs.next().expect("vector buffer too small");

        if self.max_iters == 0 {
            x.set_copy(b)
        } else {
            for _ in 0..self.max_iters {
                //// y = b - N x
                Me::aAxpby(&mut y, -1.0, &self.A.explicit(), &x, 1.0, &b);
                //// x = M^-1 y
                self.A.implicit().apply(x_, &y, &mut rest);
                //// x_k+1 = (1-w) x_k + w * x'_k
                x.acc_mul_bx(1.0 - self.omega, self.omega, &x_);
            }
        }
    }
}

pub trait Restriction<V> {
    fn restriction(&self, coarse: &mut V, fine: &V);
}

pub trait Prolongation<V> {
    fn prolongation(&self, fine: &mut V, coarse: &V);
}

pub struct FullWeightingRestriction;
impl Linear for FullWeightingRestriction {}

pub struct LinearInterpolation;
impl Linear for LinearInterpolation {}

impl Restriction<Array1<f64>> for FullWeightingRestriction {
    /// `x o x o x   =>    x x x`
    fn restriction(&self, coarse: &mut Array1<f64>, fine: &Array1<f64>) {
        #![allow(non_snake_case)]

        let N = fine.shape()[0];
        let n = coarse.shape()[0];

        assert_eq!(N, 2*n - 1);

        assert!(N > 1);

        let one_over_h_sq = 1.0 / 4.0;

        coarse[0] = one_over_h_sq * (2.0*fine[0] + fine[1]);
        for i in 1..n-1 {
            let I = 2*i;
            coarse[i] = one_over_h_sq * (fine[I-1] + 2.0*fine[I] + fine[I+1]);
        }
        coarse[n-1] = one_over_h_sq * (fine[N-2] + 2.0*fine[N-1])

    }
}

impl Prolongation<Array1<f64>> for LinearInterpolation {
    fn prolongation(&self, fine: &mut Array1<f64>, coarse: &Array1<f64>) {
        #![allow(non_snake_case)]

        let N = fine.shape()[0];
        let n = coarse.shape()[0];

        assert_eq!(N, 2*n - 1);

        assert!(N > 1);

        for i in 0..n-1 {
            let I = 2*i;
            fine[I] = coarse[i];
            fine[I+1] = 0.5*(coarse[i] + coarse[i+1]);
        }
        fine[N-1] = coarse[n-1];
    }
}

impl Restriction<Array2<f64>> for FullWeightingRestriction {
    ///
    ///```text
    /// x o x o x
    /// o o o o o        x x x
    /// x o x o x   =>   x x x
    /// o o o o o        x x x
    /// x o x o x
    /// ```
    ///
    fn restriction(&self, coarse: &mut Array2<f64>, fine: &Array2<f64>) {
        #![allow(non_snake_case)]

        let N = fine.shape()[0];
        let M = fine.shape()[1];
        let n = coarse.shape()[0];
        let m = coarse.shape()[1];

        assert_eq!(N, 2*n - 1);
        assert_eq!(M, 2*m - 1);

        // make sure that the folowing operations work
        assert!(N > 1);
        assert!(M > 1);

        // We have an implicit 0-boundary
        // The last dimension of ndarrays have the smallest stride by default (version 0.9)

        let one_over_h_xy_sq = 1.0 / 16.0;


        coarse[[0, 0]] = one_over_h_xy_sq * (
              4.*fine[[0, 0]] + 2.*fine[[0, 1]]
            + 2.*fine[[1, 0]] + 1.*fine[[1, 1]]
        );
        for j in 1..n-1 {
            let J = 2*j;

            coarse[[0,j]] = one_over_h_xy_sq * (
                  2.*fine[[0, J-1]] + 4.*fine[[0, J]] + 2.*fine[[0, J+1]]
                + 1.*fine[[1, J-1]] + 2.*fine[[1, J]] + 1.*fine[[1, J+1]]
            );
        }
        coarse[[0, m-1]] = one_over_h_xy_sq * (
              2.*fine[[0, M-2]] + 4.*fine[[0, M-1]]
            + 1.*fine[[1, M-2]] + 2.*fine[[1, M-1]]
        );


        for i in 1..n-1 {
            let I = 2*i;

            coarse[[i,0]] = one_over_h_xy_sq * (
                  2.*fine[[I-1, 0]] + 1.*fine[[I-1, 1]]
                + 4.*fine[[I  , 0]] + 2.*fine[[I  , 1]]
                + 2.*fine[[I+1, 0]] + 1.*fine[[I+1, 1]]
            );
            for j in 1..m-1 {
                let J = 2*j;

                coarse[[i,j]] = one_over_h_xy_sq * (
                      1.*fine[[I-1, J-1]] + 2.*fine[[I-1, J]] + 1.*fine[[I-1, J+1]]
                    + 2.*fine[[I  , J-1]] + 4.*fine[[I  , J]] + 2.*fine[[I  , J+1]]
                    + 1.*fine[[I+1, J-1]] + 2.*fine[[I+1, J]] + 1.*fine[[I+1, J+1]]
                );
            }
            coarse[[i,m-1]] = one_over_h_xy_sq * (
                  1.*fine[[I-1, M-2]] + 2.*fine[[I-1, M-1]]
                + 2.*fine[[I  , M-2]] + 4.*fine[[I  , M-1]]
                + 1.*fine[[I+1, M-2]] + 2.*fine[[I+1, M-1]]
            );
        }


        coarse[[n-1, 0]] = one_over_h_xy_sq * (
              2.*fine[[N-2, 0]] + 1.*fine[[N-2, 1]]
            + 4.*fine[[N-1, 0]] + 2.*fine[[N-1, 1]]
        );
        for j in 1..m-1 {
            let J = 2*j;

            coarse[[n-1,j]] = one_over_h_xy_sq * (
                  1.*fine[[N-2, J-1]] + 2.*fine[[N-2, J]] + 1.*fine[[N-2, J+1]]
                + 2.*fine[[N-1, J-1]] + 4.*fine[[N-1, J]] + 2.*fine[[N-1, J+1]]
            );
        }
        coarse[[n-1, m-1]] = one_over_h_xy_sq * (
              1.*fine[[N-2, M-2]] + 2.*fine[[N-2, M-1]]
            + 2.*fine[[N-1, M-2]] + 4.*fine[[N-1, M-1]]
        );
    }
}

impl Prolongation<Array2<f64>> for LinearInterpolation {
    ///
    ///```text
    ///              x o x o x
    /// x x x        o o o o o
    /// x x x   =>   x o x o x
    /// x x x        o o o o o
    ///              x o x o x
    /// ```
    ///
    fn prolongation(&self, fine: &mut Array2<f64>, coarse: &Array2<f64>) {
        #![allow(non_snake_case)]

        let N = fine.shape()[0];
        let M = fine.shape()[1];
        let n = coarse.shape()[0];
        let m = coarse.shape()[1];

        assert_eq!(N, 2*n - 1);
        assert_eq!(M, 2*m - 1);

        // TODO unroll
        for i in 0..n {
            let I = 2*i;

            for j in 0..m {
                let J = 2*j;
                /*
                          o o
                 x   =>   o x
                */
                if i != 0 && j != 0 {
                    fine[[I-1, J-1]] = 0.25 * (
                          coarse[[i-1, j-1]] + coarse[[i-1, j]]
                        + coarse[[i  , j-1]] + coarse[[i  , j]]
                    );
                }
                if i != 0 {
                    fine[[I-1, J]] = 0.5 * (coarse[[i-1, j]] + coarse[[i, j]]);
                }
                if j != 0 {
                    fine[[I, J-1]] = 0.5 * (coarse[[i, j-1]] + coarse[[i, j]]);
                }
                fine[[I,J]] = coarse[[i,j]];
            }
        }
    }
}

pub trait Coarse {
    type Coarse;
    fn coarse(&self) -> &Self::Coarse;
}

pub struct TwoGridMethod<Smooth,R,S,P> {
    max_iters: usize,
    smoothing: Smooth,
    restriction: R,
    solver: S,
    prolongation: P,
}
impl<Smooth,R,S,P> Linear for TwoGridMethod<Smooth,R,S,P> where
    Smooth: Linear, R: Linear, S: Linear, P: Linear {}
impl<Smooth,R,S,P> Symmetric for TwoGridMethod<Smooth,R,S,P> where
    Smooth: Symmetric, R: Symmetric, S: Symmetric, P: Symmetric {}

impl<Smooth,R,S,P,M,MC,V> SolverImpl<M,V> for TwoGridMethod<Smooth,R,S,P> where
    Smooth: SolverImpl<M,V,A=[V]>,
    S: SolverImpl<MC,V,A=[V]>,
    R: Restriction<V>,
    P: Prolongation<V>,
    M: Matrix<V> + Coarse<Coarse=MC>, MC: Matrix<V>,
    V: Vector
{
    type A = CoarseFineVecs<V>;
    fn size(&self) -> usize {
        5 + cmp::max(self.smoothing.size(), self.solver.size())
    }

    fn solve(&self, A: &M, x: &mut V, b: &V, vecs: &mut CoarseFineVecs<V>) {
        let (mut vecs_h, mut rest_h) = vecs.vec_h.split_at_mut(5);
        let (mut vecs_H, mut rest_H) = vecs.vec_H.split_at_mut(5);
        let mut vecs_h = vecs_h.iter_mut();
        let mut vecs_H = vecs_H.iter_mut();
        let mut r_h = vecs_h.next().expect("vector buffer too small");
        let mut r_H = vecs_H.next().expect("vector buffer too small");
        let mut e_H = vecs_H.next().expect("vector buffer too small");
        let mut e_h = vecs_h.next().expect("vector buffer too small");

        println!("", );
        for _ in 0..self.max_iters {
            //// x <- S x
            //M::aAxpby(&mut r_h, -1.0, A, x, 1.0, b);
            //println!("fine residual {:?}", r_h.norm_max());
            self.smoothing.solve(A, x, b, &mut rest_h);
            //// r_h = b - A x
            M::aAxpby(&mut r_h, -1.0, A, x, 1.0, b);
            println!("fine residual {:?}", r_h.norm_max());
            //// r_H' = R r_h
            self.restriction.restriction(&mut r_H, &r_h);
            //// e_H = A^-1 r_H'
            self.solver.solve(&A.coarse(), &mut e_H, r_H, &mut rest_H);
            //// e_h' = P e_H
            self.prolongation.prolongation(&mut e_h, &e_H);
            //// x1 = x1 + e_h
            x.inc_ax(1.0, e_h);
            //M::aAxpby(&mut r_h, -1.0, A, x, 1.0, b);
            //println!("fine residual {:?}", r_h.norm_max());
            //// x <- S x
            self.smoothing.solve(A, x, b, &mut rest_h);
            //M::aAxpby(&mut r_h, -1.0, A, x, 1.0, b);
            //println!("fine residual {:?}\n----", r_h.norm_max());
        }

    }
}

impl<Smooth,R,S,P,M,MC,V> Preconditioner<V> for (M,TwoGridMethod<Smooth,R,S,P>) where
    Smooth: SolverImpl<M,V,A=[V]>,
    S: SolverImpl<MC,V,A=[V]>,
    R: Restriction<V>,
    P: Prolongation<V>,
    M: Matrix<V> + Coarse<Coarse=MC>, MC: Matrix<V>,
    V: Vector
{
    type A = CoarseFineVecs<V>;
    fn size(&self) -> usize {
        self.1.size()
    }

    fn apply(&self, out: &mut V, x: &V, vecs: &mut CoarseFineVecs<V>) {
        out.set_ax(0.0, x);
        self.1.solve(&self.0, out, x, vecs)
    }
}

pub struct CoarseFineMatrix<Mf, Mc> {
    A_h: Mf,
    A_H: Mc,
}

pub struct CoarseFineVecs<V> {
    vec_h: Vec<V>,
    vec_H: Vec<V>,
}

impl<Mf, Mc, V> Matrix<V> for CoarseFineMatrix<Mf, Mc> where Mf: Matrix<V>, V: Vector {
    fn mul(out: &mut V, A: &Self, rhs: &V) {
        Mf::mul(out, &A.A_h, rhs)
    }

    fn aAxpby(out: &mut V, a: f64, A: &Self, x: &V, b: f64, y: &V) {
        Mf::aAxpby(out, a, &A.A_h, x, b, y)
    }

    fn inc_aAxpby(out: &mut V, a: f64, A: &Self, x: &V, b: f64, y: &V) {
        Mf::inc_aAxpby(out, a, &A.A_h, x, b, y)
    }
}

impl<'a, Mf, Mc> Coarse for CoarseFineMatrix<Mf, Mc> {
    type Coarse = Mc;
    fn coarse(&self) -> &Mc {
        &self.A_H
    }
}


#[cfg(test)]
mod test {
    use ::{Absolute, SolverImpl, CG, PCG};
    use super::*;
    use ndarray::{Array1, Array2, arr1, arr2};
    use stencil::Array2Stencil;
    use stencil::order2::laplace;

    #[test]
    fn restriction1d() {
        let fine = arr1(&[1.0, 2.0, 3.0, 4.0, 6.0]);
        let coarse_solution = arr1(&[1.0, 3.0, 4.0]);
        let mut coarse_computed = Array1::from_elem(3, -1.0);

        let restriction = FullWeightingRestriction;
        restriction.restriction(&mut coarse_computed, &fine);

        assert_eq!(coarse_computed, coarse_solution);
    }

    #[test]
    fn interpolation1d() {
        let coarse = arr1(&[0.0, 2.0, 4.0]);
        let fine_solution = arr1(&[0.0, 1.0, 2.0, 3.0, 4.0]);
        let mut fine_computed = Array1::from_elem(5, -1.0);

        let interpol = LinearInterpolation;
        interpol.prolongation(&mut fine_computed, &coarse);

        assert_eq!(fine_computed, fine_solution);
    }

    #[test]
    fn restriction2d() {
        let fine = arr2(&[
            [0.0, 2.0, 4.0,  6.0,  8.0],
            [1.0, 3.0, 5.0,  7.0,  9.0],
            [2.0, 4.0, 6.0,  8.0, 10.0],
            [3.0, 5.0, 7.0,  9.0, 11.0],
            [4.0, 6.0, 8.0, 10.0, 12.0],
        ]);
        let coarse_solution = arr2(&[
            [0.5625, 3.25, 4.3125],
            [2.0   , 6.0 , 7.0   ],
            [2.4375, 5.75, 6.1875],
        ]);
        let mut coarse_computed = Array2::from_elem((3,3), -1.0);

        let restriction = FullWeightingRestriction;
        restriction.restriction(&mut coarse_computed, &fine);

        assert_eq!(coarse_computed, coarse_solution);
    }

    #[test]
    fn interpolation2d() {
        let coarse = arr2(&[
            [0.0, 4.0,  8.0],
            [2.0, 6.0, 10.0],
            [4.0, 8.0, 12.0],
        ]);
        let fine_solution = arr2(&[
            [0.0, 2.0, 4.0,  6.0,  8.0],
            [1.0, 3.0, 5.0,  7.0,  9.0],
            [2.0, 4.0, 6.0,  8.0, 10.0],
            [3.0, 5.0, 7.0,  9.0, 11.0],
            [4.0, 6.0, 8.0, 10.0, 12.0],
        ]);
        let mut fine_computed = Array2::from_elem((5,5), -1.0);

        let interpol = LinearInterpolation;
        interpol.prolongation(&mut fine_computed, &coarse);

        assert_eq!(fine_computed, fine_solution);
    }

    #[test]
    fn smoothing() {
        let A = Array2Stencil::from(-laplace(0.5));
        let jacobi = AddSplittingIteration {
            max_iters: 60,
            omega: 4.0/5.0,
            A: A.split_jacobi(),
        };

        let mut x = Array2::from_shape_fn((61,61), |(i,j)| (1 + (i+j) % 2) as f64); // high frequent error
        let     x_solution = Array2::from_elem((61,61), 2.0);
        let mut b = Array2::from_elem((61,61), 0.0);
        Matrix::mul(&mut b, &A, &x_solution);

        println!("ERROR {:}", (&x_solution - &x).norm_max());
        let mut vecs = (0..Preconditioner::size(&jacobi)).map(|_| Array2::from_elem((61,61), 1.0)).collect::<Vec<_>>();
        jacobi.solve(&A, &mut x, &b, &mut vecs);
        println!("ERROR {:}", (&x_solution - &x).norm_max());

    }

    #[test]
    fn two_grid_method() {
        let A_h = Array2Stencil::from(-laplace(0.5));
        let A_H = Array2Stencil::from(-laplace(1.0));

        let jacobi_h = AddSplittingIteration {
            max_iters: 7,
            omega: 4.0 / 5.0,
            A: A_h.split_jacobi(),
        };

        let A = CoarseFineMatrix {
            A_H,
            A_h,
        };
        let tgm = TwoGridMethod {
            max_iters: 10,
            //smoothing: ::NeumannSeries::new(3, A.A_h.lower()).scale(1.0/4.0),
            smoothing: jacobi_h,
            restriction: FullWeightingRestriction,
            solver: CG::new(100, Absolute(1e-12)),//.inspect_residual(|k,r| {println!("[coarse solver] k: {:>3} r: {:.3e}",k,r)}),
            prolongation: LinearInterpolation,
        };

        let mut x = Array2::from_elem((61,61), 3.0);
        let     x_solution = Array2::from_elem((61,61), 2.0);
        let mut b = Array2::from_elem((61,61), 0.0);
        Matrix::mul(&mut b, &A, &x_solution);
        let b = b;
        let mut vecs = CoarseFineVecs {
            vec_h: (0..10).map(|_| Array2::from_elem((61,61), 1.0)).collect::<Vec<_>>(),
            vec_H: (0..10).map(|_| Array2::from_elem((31,31), 1.0)).collect::<Vec<_>>(),
        };
        tgm.solve(&A, &mut x, &b, &mut vecs);

        println!("ERROR {:.3e}", (&x_solution - &x).norm_max());
        assert!((x_solution - x).norm_max() < 1e-13);
    }

    #[test]
    fn cg_two_grid_method() {
        let A_h = Array2Stencil::from(-laplace(0.5));
        let A_H = Array2Stencil::from(-laplace(1.0));

        let jac_smooth = AddSplittingIteration {
            max_iters: 7,
            omega: 4.0 / 5.0,
            A: A_h.split_jacobi(),
        };

        let jac_solve = AddSplittingIteration {
            max_iters: 1000,
            omega: 1.0,
            A: A_h.split_jacobi(),
        };

        let A = CoarseFineMatrix {
            A_H,
            A_h,
        };

        let cg = PCG::new(
            TwoGridMethod {
                max_iters: 10,
                smoothing: jac_smooth,
                restriction: FullWeightingRestriction,
                solver: jac_solve,
                prolongation: LinearInterpolation,
            },
            CG::new(100, Absolute(1e-12)),
        );

        let mut x = Array2::from_elem((61,61), 3.0);
        let     x_solution = Array2::from_elem((61,61), 2.0);
        let mut b = Array2::from_elem((61,61), 0.0);
        Matrix::mul(&mut b, &A, &x_solution);
        let b = b;
        let mut vecs = CoarseFineVecs {
            vec_h: (0..10).map(|_| Array2::from_elem((61,61), 1.0)).collect::<Vec<_>>(),
            vec_H: (0..10).map(|_| Array2::from_elem((31,31), 1.0)).collect::<Vec<_>>(),
        };
        cg.solve(&A, &mut x, &b, &mut vecs);

        println!("ERROR {:.3e}", (&x_solution - &x).norm_max());
        assert!((x_solution - x).norm_max() < 1e-13);
    }
}
