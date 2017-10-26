
use ndarray::Array2;

pub trait Stencil3x3 {
    fn at(&self, di: isize, dj: isize) -> f64;
}

impl Stencil3x3 for Array2<f64> {
    #[inline(always)]
    fn at(&self, di: isize, dj: isize) -> f64 {
        debug_assert!(-1 <= di && di <= 1);
        debug_assert!(-1 <= dj && dj <= 1);
        self[((1+di) as usize, (1+dj) as usize)]
    }
}

pub mod order2 {
    use ndarray::{Array, Array2};

    pub fn d1(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
            0.0,  1.0, 0.0,
            0.0,  0.0, 0.0,
            0.0, -1.0, 0.0,
        ]).into_shape((3,3)).unwrap() / (2.0*h)
    }

    pub fn d2(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
            0.0, 0.0,  0.0,
            1.0, 0.0, -1.0,
            0.0, 0.0,  0.0,
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

    pub fn laplace(h: f64) -> Array2<f64> {
        Array::from_vec(vec![
            0.0, 1.0,   0.0,
            1.0, 0.0,  -1.0,
            0.0, -1.0,  0.0,
        ]).into_shape((3,3)).unwrap() / (2.0*h)
    }
}
