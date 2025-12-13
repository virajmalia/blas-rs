use num_complex::Complex;
use rayon::prelude::*;

/// General matrix-vector product: y = alpha * A * x + beta * y
pub trait Gemv {
    type Output;
    fn gemv(&self, cols: usize, alpha: Self::Output, x: &[Self::Output], beta: Self::Output, y: &mut [Self::Output], incx: usize, incy: usize);
}

impl Gemv for [f32] {
    type Output = f32;
    fn gemv(&self, cols: usize, alpha: f32, x: &[f32], beta: f32, y: &mut [f32], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f32;
                    for j in 0..cols {
                        sum += self[y_idx * cols + j] * x[j * incx];
                    }
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
    }
}

impl Gemv for [f64] {
    type Output = f64;
    fn gemv(&self, cols: usize, alpha: f64, x: &[f64], beta: f64, y: &mut [f64], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = 0.0f64;
                    for j in 0..cols {
                        sum += self[y_idx * cols + j] * x[j * incx];
                    }
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
    }
}

impl Gemv for [Complex<f32>] {
    type Output = Complex<f32>;
    fn gemv(&self, cols: usize, alpha: Complex<f32>, x: &[Complex<f32>], beta: Complex<f32>, y: &mut [Complex<f32>], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = Complex::new(0.0, 0.0);
                    for j in 0..cols {
                        sum += self[y_idx * cols + j] * x[j * incx];
                    }
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
    }
}

impl Gemv for [Complex<f64>] {
    type Output = Complex<f64>;
    fn gemv(&self, cols: usize, alpha: Complex<f64>, x: &[Complex<f64>], beta: Complex<f64>, y: &mut [Complex<f64>], incx: usize, incy: usize) {
        y.par_iter_mut()
            .enumerate()
            .for_each(|(i, y_elem)| {
                if i % incy == 0 {
                    let y_idx = i / incy;
                    let mut sum = Complex::new(0.0, 0.0);
                    for j in 0..cols {
                        sum += self[y_idx * cols + j] * x[j * incx];
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
    fn test_gemv_f32() {
        let a_f32 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_f32 = vec![1.0, 2.0, 3.0];
        let mut y_f32 = vec![0.0, 0.0];
        
        a_f32.as_slice().gemv(3, 1.0, x_f32.as_slice(), 0.0, &mut y_f32, 1, 1);
        assert_eq!(y_f32[0], 14.0);
        assert_eq!(y_f32[1], 32.0);
    }

    #[test]
    fn test_gemv_f64() {
        let a_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_f64 = vec![1.0, 2.0, 3.0];
        let mut y_f64 = vec![0.0, 0.0];
        
        a_f64.as_slice().gemv(3, 1.0, x_f64.as_slice(), 0.0, &mut y_f64, 1, 1);
        assert_eq!(y_f64[0], 14.0);
        assert_eq!(y_f64[1], 32.0);
    }

    #[test]
    fn test_gemv_complex_f32() {
        let a_complex_f32: Vec<Complex<f32>> = vec![
            Complex::new(1.0, 0.0), Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0), Complex::new(4.0, 0.0),
        ];
        let x_complex_f32: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let mut y_complex_f32: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        
        a_complex_f32.as_slice().gemv(2, Complex::new(1.0, 0.0), x_complex_f32.as_slice(), Complex::new(0.0, 0.0), &mut y_complex_f32, 1, 1);
        assert_eq!(y_complex_f32[0], Complex::new(5.0, 0.0));
        assert_eq!(y_complex_f32[1], Complex::new(11.0, 0.0));
    }
}
