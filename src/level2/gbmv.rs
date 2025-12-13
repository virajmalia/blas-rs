use num_complex::Complex;
use rayon::prelude::*;

/// General band matrix-vector product: y = alpha * A * x + beta * y
/// where A is a general band matrix
pub trait Gbmv {
    type Output;
    fn gbmv(&self, m: usize, n: usize, kl: usize, ku: usize, alpha: Self::Output, x: &[Self::Output], beta: Self::Output, y: &mut [Self::Output], incx: usize, incy: usize);
}

impl Gbmv for [f32] {
    type Output = f32;
    fn gbmv(&self, _m: usize, n: usize, kl: usize, ku: usize, alpha: f32, x: &[f32], beta: f32, y: &mut [f32], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f32;
                    for j in 0..n {
                        if y_idx <= kl + j {
                            let band_idx = kl + j - y_idx;
                            if band_idx < (kl + ku + 1) {
                                let a_idx = band_idx * n + j;
                                if a_idx < self.len() {
                                    sum += self[a_idx] * x[j * incx];
                                }
                            }
                        }
                    }
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
    }
}

impl Gbmv for [f64] {
    type Output = f64;
    fn gbmv(&self, _m: usize, n: usize, kl: usize, ku: usize, alpha: f64, x: &[f64], beta: f64, y: &mut [f64], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f64;
                    for j in 0..n {
                        if y_idx <= kl + j {
                            let band_idx = kl + j - y_idx;
                            if band_idx < (kl + ku + 1) {
                                let a_idx = band_idx * n + j;
                                if a_idx < self.len() {
                                    sum += self[a_idx] * x[j * incx];
                                }
                            }
                        }
                    }
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
    }
}

impl Gbmv for [Complex<f32>] {
    type Output = Complex<f32>;
    fn gbmv(&self, _m: usize, n: usize, kl: usize, ku: usize, alpha: Complex<f32>, x: &[Complex<f32>], beta: Complex<f32>, y: &mut [Complex<f32>], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = Complex::new(0.0, 0.0);
                    for j in 0..n {
                        let band_idx = kl + j - y_idx;
                        if band_idx < (kl + ku + 1) {
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

impl Gbmv for [Complex<f64>] {
    type Output = Complex<f64>;
    fn gbmv(&self, _m: usize, n: usize, kl: usize, ku: usize, alpha: Complex<f64>, x: &[Complex<f64>], beta: Complex<f64>, y: &mut [Complex<f64>], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = Complex::new(0.0, 0.0);
                    for j in 0..n {
                        let band_idx = kl + j - y_idx;
                        if band_idx < (kl + ku + 1) {
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
    fn test_gbmv_f32() {
        let ab_f32: Vec<f32> = vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0];
        let x_f32 = vec![1.0, 2.0];
        let mut y_f32 = vec![0.0, 0.0];
        ab_f32.as_slice().gbmv(2, 2, 0, 1, 1.0, x_f32.as_slice(), 0.0, &mut y_f32, 1, 1);
        //println!("y_f32 = {:?}", y_f32);
        assert_eq!(y_f32[0], 0.0);
        assert_eq!(y_f32[1], 2.0);
    }

    #[test]
    fn test_gbmv_f64() {
        let ab_f64 = vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0];
        let x_f64 = vec![1.0, 2.0];
        let mut y_f64 = vec![0.0, 0.0];
        ab_f64.as_slice().gbmv(2, 2, 0, 1, 1.0, x_f64.as_slice(), 0.0, &mut y_f64, 1, 1);
        //println!("y_f64 = {:?}", y_f64);
        assert_eq!(y_f64[0], 0.0);
        assert_eq!(y_f64[1], 2.0);
    }
}
