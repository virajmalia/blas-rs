use rayon::prelude::*;

/// Symmetric packed matrix-vector product: y = alpha * A * x + beta * y
/// where A is a symmetric packed matrix
pub trait Spmv {
    type Output;
    fn spmv(&self, n: usize, alpha: Self::Output, x: &[Self::Output], beta: Self::Output, y: &mut [Self::Output], incx: usize, incy: usize);
}

impl Spmv for [f32] {
    type Output = f32;
    fn spmv(&self, n: usize, alpha: f32, x: &[f32], beta: f32, y: &mut [f32], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f32;
                    let mut k = y_idx * (y_idx + 1) / 2;
                    for j in 0..n {
                        if j <= y_idx {
                            sum += self[k] * x[j * incx];
                            k += 1;
                        } else {
                            let idx = j * (j + 1) / 2 + y_idx;
                            if idx < self.len() {
                                sum += self[idx] * x[j * incx];
                            }
                        }
                    }
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
    }
}

impl Spmv for [f64] {
    type Output = f64;
    fn spmv(&self, n: usize, alpha: f64, x: &[f64], beta: f64, y: &mut [f64], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f64;
                    let mut k = y_idx * (y_idx + 1) / 2;
                    for j in 0..n {
                        if j <= y_idx {
                            sum += self[k] * x[j * incx];
                            k += 1;
                        } else {
                            let idx = j * (j + 1) / 2 + y_idx;
                            if idx < self.len() {
                                sum += self[idx] * x[j * incx];
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
    fn test_spmv_f32() {
        let ap_f32 = vec![1.0, 2.0, 3.0];
        let x_f32 = vec![1.0, 2.0];
        let mut y_f32 = vec![0.0, 0.0];
        
        ap_f32.as_slice().spmv(2, 1.0, x_f32.as_slice(), 0.0, &mut y_f32, 1, 1);
    }

    #[test]
    fn test_spmv_f64() {
        let ap_f64 = vec![1.0, 2.0, 3.0];
        let x_f64 = vec![1.0, 2.0];
        let mut y_f64 = vec![0.0, 0.0];
        
        ap_f64.as_slice().spmv(2, 1.0, x_f64.as_slice(), 0.0, &mut y_f64, 1, 1);
    }
}
