use ndarray::{Array1, Array2};
use marke;

pub trait Restriction<V> {
    fn restriction(&self, coarse: &mut V, fine: &V);
}

pub trait Prolongation<V> {
    fn prolongation(&self, fine: &mut V, coarse: &V);
}

pub struct FullWeightingRestriction;
impl marke::Linear for FullWeightingRestriction {}

pub struct LinearInterpolation;
impl marke::Linear for LinearInterpolation {}

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


#[cfg(test)]
mod test {
    use super::*;
    use ndarray::{Array1, Array2, arr1, arr2};

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

}
