use rayon::prelude::*;

/// Symmetric matrix-vector product: y = alpha * A * x + beta * y
/// where A is symmetric
pub trait Symv {
    type Output;
    fn symv(&self, n: usize, alpha: Self::Output, x: &[Self::Output], beta: Self::Output, y: &mut [Self::Output], incx: usize, incy: usize);
}

impl Symv for [f32] {
    type Output = f32;
    fn symv(&self, n: usize, alpha: f32, x: &[f32], beta: f32, y: &mut [f32], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        let a_idx = y_idx * n + j;
                        if a_idx < self.len() {
                            sum += self[a_idx] * x[j * incx];
                        }
                    }
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
    }
}

impl Symv for [f64] {
    type Output = f64;
    fn symv(&self, n: usize, alpha: f64, x: &[f64], beta: f64, y: &mut [f64], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f64;
                    for j in 0..n {
                        let a_idx = y_idx * n + j;
                        if a_idx < self.len() {
                            sum += self[a_idx] * x[j * incx];
                        }
                    }
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symv_f32() {
        let a_f32 = vec![1.0, 2.0, 2.0, 3.0];
        let x_f32 = vec![1.0, 2.0];
        let mut y_f32 = vec![0.0, 0.0];
        
        a_f32.as_slice().symv(2, 1.0, x_f32.as_slice(), 0.0, &mut y_f32, 1, 1);
        assert_eq!(y_f32[0], 5.0);
        assert_eq!(y_f32[1], 8.0);
    }

    #[test]
    fn test_symv_f64() {
        let a_f64 = vec![1.0, 2.0, 2.0, 3.0];
        let x_f64 = vec![1.0, 2.0];
        let mut y_f64 = vec![0.0, 0.0];
        
        a_f64.as_slice().symv(2, 1.0, x_f64.as_slice(), 0.0, &mut y_f64, 1, 1);
        assert_eq!(y_f64[0], 5.0);
        assert_eq!(y_f64[1], 8.0);
    }
}
