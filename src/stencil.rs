use ndarray::{Array1, Array2};

use ::{Matrix, Preconditioner};
use multigrid::AdditiveSplitMatrix;

#[derive(Clone, Debug)]
pub struct Array2Stencil(Array2<f64>);

impl Array2Stencil {
    pub fn from(arr: Array2<f64>) -> Array2Stencil {
        assert_eq!(arr.shape()[0], 3);
        assert_eq!(arr.shape()[1], 3);
        Array2Stencil(arr)
    }
}

#[repr(usize)]
pub enum IxI {
    #[allow(non_camel_case_types)]
    im1 = 0,
    #[allow(non_camel_case_types)]
    i = 1,
    #[allow(non_camel_case_types)]
    ip1 = 2,
}
#[repr(usize)]
pub enum IxJ {
    #[allow(non_camel_case_types)]
    jm1 = 0,
    #[allow(non_camel_case_types)]
    j = 1,
    #[allow(non_camel_case_types)]
    jp1 = 2,
}

pub trait Stencil3x3 {
    fn at(&self, di: IxI, dj: IxJ) -> f64;
}

impl Array2Stencil {
    // y <- Stencil * rhs
    // out_ij <- f(i, j, y_ij)
    fn mul_f<F>(&self, rhs: &Array2<f64>, f: F, out: &mut Array2<f64>) where F: Fn(usize, usize, f64) -> f64 {
        let n = rhs.shape()[0];
        let m = rhs.shape()[1];
        assert_eq!(n, out.shape()[0]);
        assert_eq!(m, out.shape()[1]);

        let Aim1jm1 = self.at(IxI::im1, IxJ::jm1);
        let Ai__jm1 = self.at(IxI::i  , IxJ::jm1);
        let Aip1jm1 = self.at(IxI::ip1, IxJ::jm1);
        let Aim1j__ = self.at(IxI::im1, IxJ::j  );
        let Ai__j__ = self.at(IxI::i  , IxJ::j  );
        let Aip1j__ = self.at(IxI::ip1, IxJ::j  );
        let Aim1jp1 = self.at(IxI::im1, IxJ::jp1);
        let Ai__jp1 = self.at(IxI::i  , IxJ::jp1);
        let Aip1jp1 = self.at(IxI::ip1, IxJ::jp1);


        out[[0, 0]] = f(0, 0,
              Ai__j__*rhs[[0,0]] + Ai__jp1*rhs[[0,1]]
            + Aip1j__*rhs[[1,0]] + Aip1jp1*rhs[[1,1]]
        );
        for j in 1..m-1 {
            out[[0,j]] = f(0, j,
                  Ai__jm1*rhs[[0,j-1]] + Ai__j__*rhs[[0,j]] + Ai__jp1*rhs[[0,j+1]]
                + Aip1jm1*rhs[[1,j-1]] + Aip1j__*rhs[[1,j]] + Aip1jp1*rhs[[1,j+1]]
            );
        }
        out[[0, m-1]] = f(0, m-1,
              Ai__jm1*rhs[[0,m-2]] + Ai__j__*rhs[[0,m-1]]
            + Aip1jm1*rhs[[1,m-2]] + Aip1j__*rhs[[1,m-1]]
        );


        for i in 1..n-1 {
            out[[i, 0]] = f(i, 0,
                  Aim1j__*rhs[[i-1,0]] + Aim1jp1*rhs[[i-1,1]]
                + Ai__j__*rhs[[i  ,0]] + Ai__jp1*rhs[[i  ,1]]
                + Aip1j__*rhs[[i+1,0]] + Aip1jp1*rhs[[i+1,1]]
            );
            for j in 1..m-1 {
                out[[i, j]] = f(i, j,
                      Aim1jm1*rhs[[i-1,j-1]] + Aim1j__*rhs[[i-1,j]] + Aim1jp1*rhs[[i-1,j+1]]
                    + Ai__jm1*rhs[[i  ,j-1]] + Ai__j__*rhs[[i  ,j]] + Ai__jp1*rhs[[i  ,j+1]]
                    + Aip1jm1*rhs[[i+1,j-1]] + Aip1j__*rhs[[i+1,j]] + Aip1jp1*rhs[[i+1,j+1]]
                );
            }
            out[[i, m-1]] = f(i, m-1,
                  Aim1jm1*rhs[[i-1,m-2]] + Aim1j__*rhs[[i-1,m-1]]
                + Ai__jm1*rhs[[i  ,m-2]] + Ai__j__*rhs[[i  ,m-1]]
                + Aip1jm1*rhs[[i+1,m-2]] + Aip1j__*rhs[[i+1,m-1]]
            );
        }


        out[[n-1, 0]] = f(n-1, 0,
              Aim1j__*rhs[[n-2,0]] + Aim1jp1*rhs[[n-2,1]]
            + Ai__j__*rhs[[n-1,0]] + Ai__jp1*rhs[[n-1,1]]
        );
        for j in 1..m-1 {
            out[[n-1,j]] = f(n-1, j,
                  Aim1jm1*rhs[[n-2,j-1]] + Aim1j__*rhs[[n-2,j]] + Aim1jp1*rhs[[n-2,j+1]]
                + Ai__jm1*rhs[[n-1,j-1]] + Ai__j__*rhs[[n-1,j]] + Ai__jp1*rhs[[n-1,j+1]]
            );
        }
        out[[n-1, m-1]] = f(n-1, m-1,
              Aim1jm1*rhs[[n-2,m-2]] + Aim1j__*rhs[[n-2,m-1]]
            + Ai__jm1*rhs[[n-1,m-2]] + Ai__j__*rhs[[n-1,m-1]]
        );
    }

    pub fn diag(&self) -> Array2Stencil {
        Array2Stencil(Array1::from_vec(vec![
            0.0, 0.0, 0.0,
            0.0, self.at(IxI::i, IxJ::j), 0.0,
            0.0, 0.0, 0.0,
        ]).into_shape((3,3)).unwrap())
    }

    pub fn diag_inv(&self) -> Array2Stencil {
        Array2Stencil(Array1::from_vec(vec![
            0.0, 0.0, 0.0,
            0.0, 1.0/self.at(IxI::i, IxJ::j), 0.0,
            0.0, 0.0, 0.0,
        ]).into_shape((3,3)).unwrap())
    }

    /// Assuming i-major: index = i * stride + j
    pub fn lower_diag(&self) -> Array2Stencil {
        let mut l = self.clone();
        // set all indices higher than (i,j) to zero
        l.0[[1,2]] = 0.0;
        l.0[[2,0]] = 0.0;
        l.0[[2,1]] = 0.0;
        l.0[[2,2]] = 0.0;
        l
    }

    /// Assuming i-major: index = i * stride + j
    pub fn lower(&self) -> Array2Stencil {
        let mut l = self.lower_diag();
        l.0[[1,1]] = 0.0;
        l
    }

    pub fn upper(&self) -> Array2Stencil {
        Array2Stencil(self.0.clone() - self.lower_diag().0)
    }

    pub fn off_diag(&self) -> Array2Stencil {
        let mut l = self.clone();
        l.0[[1,1]] = 0.0;
        l
    }

    pub fn split_jacobi(&self) -> AdditiveSplitMatrix<f64, Array2Stencil> {
        AdditiveSplitMatrix {
            implicit: self.at(IxI::i, IxJ::j),
            explicit: self.off_diag(),
        }
    }

    pub fn split_gauss_seidl(&self) -> AdditiveSplitMatrix<Array2Stencil, Array2Stencil> {
        AdditiveSplitMatrix {
            implicit: self.lower_diag(),
            explicit: self.upper(),
        }
    }

}

impl Stencil3x3 for Array2Stencil {
    #[inline(always)]
    fn at(&self, di: IxI, dj: IxJ) -> f64 {
        self.0[(di as usize, dj as usize)]
    }
}

impl Matrix<Array2<f64>> for Array2Stencil {
    fn mul(out: &mut Array2<f64>, A: &Array2Stencil, rhs: &Array2<f64>) {
        A.mul_f(rhs, |_,_,val| val, out)
    }

    fn aAxpby(out: &mut Array2<f64>, a: f64, A: &Array2Stencil, x: &Array2<f64>, b: f64, y: &Array2<f64>) {
        A.mul_f(x, |i,j,val| a*val + b * y[[i,j]], out)
    }

    fn inc_aAxpby(out: &mut Array2<f64>, a: f64, A: &Array2Stencil, x: &Array2<f64>, b: f64, y: &Array2<f64>) {
        unimplemented!() // TODO improve mul_f to allow for inc
    }
}

impl Preconditioner<Array2<f64>> for Array2Stencil {
    type A = [Array2<f64>];
    fn size(&self) -> usize {
        0
    }

    fn apply(&self, out: &mut Array2<f64>, x: &Array2<f64>, _:&mut [Array2<f64>]) {
        Matrix::mul(out, &self, x)
    }
}

#[cfg(test)]
mod test {
    use ::{Absolute, CG, SolverImpl, Vector};
    use super::*;
    use super::order2::{d1d2,div,laplace};
    use ndarray::Array;

    #[test]
    fn diag_and_lower() {
        let s = Array2Stencil(2.0*div(0.5) + d1d2(0.5) + laplace(0.5));
        assert_eq!(s.at(IxI::im1, IxJ::jm1), 1.0);
        assert_eq!(s.at(IxI::im1, IxJ::j), -1.0);
        assert_eq!(s.at(IxI::im1, IxJ::jp1), -1.0);
        assert_eq!(s.at(IxI::i, IxJ::jm1), -1.0);
        assert_eq!(s.at(IxI::i, IxJ::j), -4.0);
        assert_eq!(s.at(IxI::i, IxJ::jp1), 3.0);
        assert_eq!(s.at(IxI::ip1, IxJ::jm1), -1.0);
        assert_eq!(s.at(IxI::ip1, IxJ::j), 3.0);
        assert_eq!(s.at(IxI::ip1, IxJ::jp1), 1.0);
        let diag = s.diag();
        assert_eq!(diag.at(IxI::im1, IxJ::jm1), 0.0);
        assert_eq!(diag.at(IxI::im1, IxJ::j), 0.0);
        assert_eq!(diag.at(IxI::im1, IxJ::jp1), 0.0);
        assert_eq!(diag.at(IxI::i, IxJ::jm1), 0.0);
        assert_eq!(diag.at(IxI::i, IxJ::j), -4.0);
        assert_eq!(diag.at(IxI::i, IxJ::jp1), 0.0);
        assert_eq!(diag.at(IxI::ip1, IxJ::jm1), 0.0);
        assert_eq!(diag.at(IxI::ip1, IxJ::j), 0.0);
        assert_eq!(diag.at(IxI::ip1, IxJ::jp1), 0.0);
        let lower = s.lower();
        assert_eq!(lower.at(IxI::im1, IxJ::jm1), 1.0);
        assert_eq!(lower.at(IxI::im1, IxJ::j), -1.0);
        assert_eq!(lower.at(IxI::im1, IxJ::jp1), -1.0);
        assert_eq!(lower.at(IxI::i, IxJ::jm1), -1.0);
        assert_eq!(lower.at(IxI::i, IxJ::j), 0.0);
        assert_eq!(lower.at(IxI::i, IxJ::jp1), 0.0);
        assert_eq!(lower.at(IxI::ip1, IxJ::jm1), 0.0);
        assert_eq!(lower.at(IxI::ip1, IxJ::j), 0.0);
        assert_eq!(lower.at(IxI::ip1, IxJ::jp1), 0.0);
    }

    #[test]
    fn stencil_cg() {
        let n = 31;
        let A = Array2Stencil::from(-laplace(1.0));
        let mut x = Array2::from_elem((n,n), 3.0);
        let mut x_solution = Array2::from_elem((n,n), 1.0);
        let mut b = Array2::from_elem((n,n), 0.0);
        Matrix::mul(&mut b, &A, &x_solution);
        let cg = CG::new(100, Absolute(1e-12)).inspect_residual(|k,r| {println!("k: {:>3} r: {:.3e}",k,r)});
        let mut vecs = (0..10).map(|_| Array2::from_elem((n,n), 1.0)).collect::<Vec<_>>();
        cg.solve(&A, &mut x, &b, &mut vecs);

        println!("ERROR {:.3e}", (&x_solution - &x).norm_max());
        assert!((x_solution - x).norm_max() < 1e-12);
    }
}

pub mod order2 {
    use ndarray::{Array, Array2};

    pub fn d1(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
            0.0, -1.0, 0.0,
            0.0,  0.0, 0.0,
            0.0,  1.0, 0.0,
        ]).into_shape((3,3)).unwrap() / (2.0*h)
    }

    pub fn d2(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
             0.0, 0.0, 0.0,
            -1.0, 0.0, 1.0,
             0.0, 0.0, 0.0,
        ]).into_shape((3,3)).unwrap() / (2.0*h)
    }

    pub fn d1d1(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
            0.0,  1.0, 0.0,
            0.0, -2.0, 0.0,
            0.0,  1.0, 0.0,
        ]).into_shape((3,3)).unwrap() / (4.0*h*h)
    }

    pub fn d2d2(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
            0.0,  0.0, 0.0,
            1.0, -2.0, 1.0,
            0.0,  0.0, 0.0,
        ]).into_shape((3,3)).unwrap() / (4.0*h*h)
    }


    pub fn d1d2(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
             1.0, 0.0, -1.0,
             0.0, 0.0,  0.0,
            -1.0, 0.0,  1.0,
        ]).into_shape((3,3)).unwrap() / (4.0*h*h)
    }

    pub fn div(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
             0.0, -1.0,  0.0,
            -1.0, 0.0,  1.0,
             0.0,  1.0, 0.0,
        ]).into_shape((3,3)).unwrap() / (2.0*h)
    }

    pub fn laplace(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
            0.0,  1.0, 0.0,
            1.0, -4.0, 1.0,
            0.0,  1.0, 0.0,
        ]).into_shape((3,3)).unwrap() / (4.0*h*h)
    }

    #[cfg(test)]
    mod test {
        use super::*;

        #[test]
        fn derivatives() {
            let dx = d1(0.5);
            assert_eq!(dx[[1,1]], 0.0); // i,j
            assert_eq!(dx[[0,1]], -1.0); // i-1,j
            assert_eq!(dx[[2,1]], 1.0); // i+1,j
            let dy = d2(0.5);
            assert_eq!(dy[[1,1]], 0.0); // i,j
            assert_eq!(dy[[1,0]], -1.0); // i-1,j
            assert_eq!(dy[[1,2]], 1.0); // i+1,j
            let d11 = d1d1(0.5);
            assert_eq!(d11[[1,1]], -2.0); // i,j
            assert_eq!(d11[[0,1]], 1.0); // i-1,j
            assert_eq!(d11[[2,1]], 1.0); // i+1,j

            assert_eq!(div(0.5), d1(0.5) + d2(0.5));
            assert_eq!(laplace(0.5), d1d1(0.5) + d2d2(0.5));
        }
    }
}
