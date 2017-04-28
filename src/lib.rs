

extern crate ndarray;

use ndarray::{Array, Array1, Array2};

pub mod stencil;

pub mod lib2;

pub fn zeros(len: usize) -> Array1<f64> {
    Array::from_elem(len, 0.0)
}

pub fn ones(len: usize) -> Array1<f64> {
    Array::from_elem(len, 1.0)
}

pub fn dot(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.dot(b)
}

pub fn norm2sq(v: &Array1<f64>) -> f64 {
    v.mapv(|vi| vi*vi).scalar_sum()
}

// TODO
fn isqrt(i: usize) -> usize {
    let s = (i as f64).sqrt() as usize;
    if s*s == i {
        s
    } else if (s-1)*(s-1) == i {
        s-1
    } else if (s+1)*(s+1) == i {
        s+1
    } else {
        panic!()
    }
}

pub trait LinOp<T> {
    fn mul(&self, rhs: &T) -> T;
}

impl<'a> LinOp<Array1<f64>> for &'a Array2<f64> {
    fn mul(&self, rhs: &Array1<f64>) -> Array1<f64> {
        self.dot(rhs)
    }
}

/// a,b: S -- Scalar eg: f64
/// x,y: V -- Vector eg: Array1<f64> or even Array2<f64>
/// A  : M -- Matrix eg: Array2<f64> or NegLaplace2D
#[allow(non_snake_case)]
pub trait MatVec<S, V, M> {
    fn dot(x: &V, y: &V) -> S;
    /// out <- Ax
    fn mul(A: &M, x: &V, out: &mut V);
    /// x <- x + by
    fn accx_x_p_by(acc: &mut V, b: S, y: &V);
    /// out <- ax + y
    fn ax_p_y(a: S, x: &V, y: &V, out: &mut V);
    /// out <- aAx + by
    fn aAx_p_by(a: S, A: &M, x: &V, b: S, y: &V, out: &mut V);
}

#[allow(non_snake_case)]
impl<M> MatVec<f64, Array1<f64>, M> for M where M: LinOp<Array1<f64>> {
    fn dot(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
        x.dot(&y)
    }

    // TODO sub-optimal
    fn mul(A: &M, x: &Array1<f64>, out: &mut Array1<f64>) {
        *out = A.mul(&x);
    }

    // TODO sub-optimal
    fn accx_x_p_by(acc: &mut Array1<f64>, b: f64, y: &Array1<f64>) {
        *acc += &(b*y);
    }

    // TODO sub-optimal
    fn ax_p_y(a: f64, x: &Array1<f64>, y: &Array1<f64>, out: &mut Array1<f64>) {
        *out = a*x + y;
    }

    // TODO sub-optimal
    fn aAx_p_by(a: f64, A: &M, x: &Array1<f64>, b: f64, y: &Array1<f64>, out: &mut Array1<f64>) {
        *out = a*A.mul(&x) + b*y;
    }
}


/// nxn
pub trait Matrix {
    fn mul_store(&self, rhs: &Array1<f64>, out: &mut Array1<f64>);
    fn mul(&self, rhs: &Array1<f64>) -> Array1<f64> {
        let n = rhs.len();
        let mut out = zeros(n);
        self.mul_store(rhs, &mut out);
        out
    }
}

impl Matrix for Array2<f64> {
    fn mul_store(&self, rhs: &Array1<f64>, out: &mut Array1<f64>) {
        *out = self.dot(rhs)
    }
}

pub struct MatrixLaplace2D{ }

impl MatrixLaplace2D {
    pub fn new() -> MatrixLaplace2D {
        MatrixLaplace2D{}
    }
}

impl Matrix for MatrixLaplace2D {
    fn mul_store(&self, rhs: &Array1<f64>, out: &mut Array1<f64>) {
        let n = isqrt(rhs.len());
        // stride
        let sx = 1;
        let sy = n;

        // neg-x neg-y corner: no i-1, j-1
        let i_j = 0;
        let i_jp1 = i_j + sy;
        let ip1_j = i_j + sx;
        out[i_j] = -4.0 * rhs[i_j] + rhs[ip1_j] + rhs[i_jp1];

        // neg-y border: no j-1
        for i in 1..n-1 {
            let i_j = i*sx + 0*sy;
            let i_jp1 = i_j + sy;
            let im1_j = i_j - sx;
            let ip1_j = i_j + sx;
            out[i_j] = rhs[im1_j] - 4.0 * rhs[i_j] + rhs[ip1_j] + rhs[i_jp1];
        }

        // pos-x neg-y corner: no i+1, j-1
        let i_j = (n-1)*sx + 0*sy;
        let i_jp1 = i_j + sy;
        let im1_j = i_j - sx;
        out[i_j] = rhs[im1_j] - 4.0 * rhs[i_j] + rhs[i_jp1];


        for j in 1..n-1 {
            // neg-x border: no i-1
            let i_j = 0*sx + j*sy;
            let i_jm1 = i_j - sy;
            let i_jp1 = i_j + sy;
            let ip1_j = i_j + sx;
            out[i_j] = rhs[i_jm1] - 4.0 * rhs[i_j] + rhs[ip1_j] + rhs[i_jp1];

            // apply stencil to all inner points
            for i in 1..n-1 {
                let i_j = i*sx + j*sy;
                let i_jm1 = i_j - sy;
                let i_jp1 = i_j + sy;
                let im1_j = i_j - sx;
                let ip1_j = i_j + sx;
                out[i_j] = rhs[i_jm1] + rhs[im1_j] - 4.0 * rhs[i_j] + rhs[ip1_j] + rhs[i_jp1];
            }

            // pos-x border: no i+1
            let i_j = (n-1)*sx + j*sy;
            let i_jm1 = i_j - sy;
            let i_jp1 = i_j + sy;
            let im1_j = i_j - sx;
            out[i_j] = rhs[i_jm1] + rhs[im1_j] - 4.0 * rhs[i_j] + rhs[i_jp1];
        }


        // neg-x pos-y corner: no i-1, j+1
        let i_j = 0*sx + (n-1)*sy;
        let i_jm1 = i_j - sy;
        let ip1_j = i_j + sx;
        out[i_j] = rhs[i_jm1] - 4.0 * rhs[i_j] + rhs[ip1_j];

        // pos-y border: no j+1
        for i in 1..n-1 {
            let i_j = i*sx + (n-1)*sy;
            let i_jm1 = i_j - sy;
            let im1_j = i_j - sx;
            let ip1_j = i_j + sx;
            out[i_j] = rhs[i_jm1] + rhs[im1_j] - 4.0 * rhs[i_j] + rhs[ip1_j];
        }

        // pos-x pos-y corner: no i+1, j+1
        let i_j = (n-1)*sx + (n-1)*sy;
        let i_jm1 = i_j - sy;
        let im1_j = i_j - sx;
        out[i_j] = rhs[i_jm1] + rhs[im1_j] - 4.0 * rhs[i_j];
    }
}

pub struct Identity { }

impl Identity {
    pub fn new() -> Identity {
        Identity{}
    }
}

impl Matrix for Identity {
    fn mul_store(&self, rhs: &Array1<f64>, out: &mut Array1<f64>) {
        *out = rhs.clone()
    }
}


use stencil::Stencil3x3;

pub struct Stencil3x3Matrix {
    stencil: Array2<f64>,
}

impl Stencil3x3Matrix {
    pub fn new(stencil: Array2<f64>) -> Stencil3x3Matrix {
        assert_eq!(stencil.shape(), &[3,3]);
        Stencil3x3Matrix{stencil:stencil}
    }
}

impl Matrix for Stencil3x3Matrix {
    fn mul_store(&self, rhs: &Array1<f64>, out: &mut Array1<f64>) {
        let n = isqrt(rhs.len());
        // stride
        let sx = 1;
        let sy = n;

        for j in 0..n {
            for i in 0..n {
                let i_j = i*sx + j*sy;

                // TODO slow 3x3 stencil?

                if j != 0 {
                    if i != 0 {
                        out[i_j] = self.stencil.at(-1,-1) * rhs[i_j - sx - sy];
                    }
                    out[i_j] = self.stencil.at(0,-1) * rhs[i_j - sy];
                    if i != n-1 {
                        out[i_j] = self.stencil.at(1,-1) * rhs[i_j + sx - sy];
                    }
                }

                if i != 0 {
                    out[i_j] = self.stencil.at(-1,0) * rhs[i_j - sx];
                }
                out[i_j] = self.stencil.at(0,0) * rhs[i_j];
                if i != n-1 {
                    out[i_j] = self.stencil.at(1,0) * rhs[i_j + sx];
                }

                if j != n-1 {
                    if i != 0 {
                        out[i_j] = self.stencil.at(-1,1) * rhs[i_j - sx + sy];
                    }
                    out[i_j] = self.stencil.at(0,1) * rhs[i_j + sy];
                    if i != n-1 {
                        out[i_j] = self.stencil.at(1,1) * rhs[i_j + sx + sy];
                    }
                }

            }
        }
    }
}
