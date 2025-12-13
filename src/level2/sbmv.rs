use rayon::prelude::*;

/// Symmetric band matrix-vector product: y = alpha * A * x + beta * y
/// where A is a symmetric band matrix
pub trait Sbmv {
    type Output;
    fn sbmv(&self, n: usize, k: usize, alpha: Self::Output, x: &[Self::Output], beta: Self::Output, y: &mut [Self::Output], incx: usize, incy: usize);
}

impl Sbmv for [f32] {
    type Output = f32;
    fn sbmv(&self, n: usize, k: usize, alpha: f32, x: &[f32], beta: f32, y: &mut [f32], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        let band_idx = k + j - y_idx;
                        if band_idx < (2 * k + 1) {
                            let a_idx = band_idx * n + j;
                            if a_idx < self.len() {
                                sum += self[a_idx] * x[j * incx];
                            }
                        }
                    }
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
    }
}

impl Sbmv for [f64] {
    type Output = f64;
    fn sbmv(&self, n: usize, k: usize, alpha: f64, x: &[f64], beta: f64, y: &mut [f64], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f64;
                    for j in 0..n {
                        let band_idx = k + j - y_idx;
                        if band_idx < (2 * k + 1) {
                            let a_idx = band_idx * n + j;
                            if a_idx < self.len() {
                                sum += self[a_idx] * x[j * incx];
                            }
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
    fn test_sbmv_f32() {
        let ab_f32 = vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0];
        let x_f32 = vec![1.0, 2.0];
        let mut y_f32 = vec![0.0, 0.0];
        
        ab_f32.as_slice().sbmv(2, 1, 1.0, x_f32.as_slice(), 0.0, &mut y_f32, 1, 1);
    }

    #[test]
    fn test_sbmv_f64() {
        let ab_f64 = vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0];
        let x_f64 = vec![1.0, 2.0];
        let mut y_f64 = vec![0.0, 0.0];
        
        ab_f64.as_slice().sbmv(2, 1, 1.0, x_f64.as_slice(), 0.0, &mut y_f64, 1, 1);
    }
}
