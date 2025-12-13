use rayon::prelude::*;

/// Rank-1 update of a symmetric matrix: A := alpha * x * y^T + A
pub trait Syr {
    type Output;
    fn syr(&mut self, n: usize, alpha: Self::Output, x: &[Self::Output], incx: usize);
}

impl Syr for [f32] {
    type Output = f32;
    fn syr(&mut self, n: usize, alpha: f32, x: &[f32], incx: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < n && col < n {
                    *a_elem += alpha * x[row * incx] * x[col * incx];
                }
            });
    }
}

impl Syr for [f64] {
    type Output = f64;
    fn syr(&mut self, n: usize, alpha: f64, x: &[f64], incx: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < n && col < n {
                    *a_elem += alpha * x[row * incx] * x[col * incx];
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syr_f32() {
        let mut a_f32 = vec![0.0, 0.0, 0.0, 0.0];
        let x_f32 = vec![1.0, 2.0];
        
        a_f32.as_mut_slice().syr(2, 1.0, x_f32.as_slice(), 1);
        assert_eq!(a_f32, vec![1.0, 2.0, 2.0, 4.0]);
    }

    #[test]
    fn test_syr_f64() {
        let mut a_f64 = vec![0.0, 0.0, 0.0, 0.0];
        let x_f64 = vec![1.0, 2.0];
        
        a_f64.as_mut_slice().syr(2, 1.0, x_f64.as_slice(), 1);
        assert_eq!(a_f64, vec![1.0, 2.0, 2.0, 4.0]);
    }
}
