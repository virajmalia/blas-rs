use rayon::prelude::*;

/// Rank-2 update of a symmetric matrix: A := alpha * x * y^T + alpha * y * x^T + A
pub trait Syr2 {
    type Output;
    fn syr2(&mut self, n: usize, alpha: Self::Output, x: &[Self::Output], y: &[Self::Output], incx: usize, incy: usize);
}

impl Syr2 for [f32] {
    type Output = f32;
    fn syr2(&mut self, n: usize, alpha: f32, x: &[f32], y: &[f32], incx: usize, incy: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < n && col < n {
                    *a_elem += alpha * (x[row * incx] * y[col * incy] + y[row * incy] * x[col * incx]);
                }
            });
    }
}

impl Syr2 for [f64] {
    type Output = f64;
    fn syr2(&mut self, n: usize, alpha: f64, x: &[f64], y: &[f64], incx: usize, incy: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < n && col < n {
                    *a_elem += alpha * (x[row * incx] * y[col * incy] + y[row * incy] * x[col * incx]);
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syr2_f32() {
        let mut a_f32 = vec![0.0, 0.0, 0.0, 0.0];
        let x_f32 = vec![1.0, 2.0];
        let y_f32 = vec![1.0, 2.0];
        
        a_f32.as_mut_slice().syr2(2, 1.0, x_f32.as_slice(), y_f32.as_slice(), 1, 1);
        assert_eq!(a_f32, vec![2.0, 4.0, 4.0, 8.0]);
    }

    #[test]
    fn test_syr2_f64() {
        let mut a_f64 = vec![0.0, 0.0, 0.0, 0.0];
        let x_f64 = vec![1.0, 2.0];
        let y_f64 = vec![1.0, 2.0];
        
        a_f64.as_mut_slice().syr2(2, 1.0, x_f64.as_slice(), y_f64.as_slice(), 1, 1);
        assert_eq!(a_f64, vec![2.0, 4.0, 4.0, 8.0]);
    }
}
