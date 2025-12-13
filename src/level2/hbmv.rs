use num_complex::Complex;
use rayon::prelude::*;

/// Hermitian band matrix-vector product: y = alpha * A * x + beta * y
/// where A is a Hermitian band matrix
pub trait Hbmv {
    type Output;
    fn hbmv(&self, n: usize, k: usize, alpha: Self::Output, x: &[Self::Output], beta: Self::Output, y: &mut [Self::Output], incx: usize, incy: usize);
}

impl Hbmv for [Complex<f32>] {
    type Output = Complex<f32>;
    fn hbmv(&self, n: usize, k: usize, alpha: Complex<f32>, x: &[Complex<f32>], beta: Complex<f32>, y: &mut [Complex<f32>], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = Complex::new(0.0, 0.0);
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

impl Hbmv for [Complex<f64>] {
    type Output = Complex<f64>;
    fn hbmv(&self, n: usize, k: usize, alpha: Complex<f64>, x: &[Complex<f64>], beta: Complex<f64>, y: &mut [Complex<f64>], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = Complex::new(0.0, 0.0);
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
    use num_complex::Complex;

    #[test]
    fn test_hbmv_complex_f32() {
        let ab: Vec<Complex<f32>> = vec![
            Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0), Complex::new(3.0, 0.0),
        ];
        let x: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let mut y: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        
        ab.as_slice().hbmv(2, 1, Complex::new(1.0, 0.0), x.as_slice(), Complex::new(0.0, 0.0), &mut y, 1, 1);
    }

    #[test]
    fn test_hbmv_complex_f64() {
        let ab: Vec<Complex<f64>> = vec![
            Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0), Complex::new(3.0, 0.0),
        ];
        let x: Vec<Complex<f64>> = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let mut y: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        
        ab.as_slice().hbmv(2, 1, Complex::new(1.0, 0.0), x.as_slice(), Complex::new(0.0, 0.0), &mut y, 1, 1);
    }
}
