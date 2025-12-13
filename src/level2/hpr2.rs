use num_complex::Complex;

/// Rank-2 update of a Hermitian packed matrix: A := alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
pub trait Hpr2 {
    type Output;
    fn hpr2(&mut self, n: usize, alpha: Self::Output, x: &[Self::Output], y: &[Self::Output], incx: usize, incy: usize);
}

impl Hpr2 for [Complex<f32>] {
    type Output = Complex<f32>;
    fn hpr2(&mut self, n: usize, alpha: Complex<f32>, x: &[Complex<f32>], y: &[Complex<f32>], incx: usize, incy: usize) {
        let mut k = 0;
        for i in 0..n {
            for j in 0..=i {
                self[k] += alpha * x[i * incx] * y[j * incy].conj() + alpha.conj() * y[i * incy] * x[j * incx].conj();
                k += 1;
            }
        }
    }
}

impl Hpr2 for [Complex<f64>] {
    type Output = Complex<f64>;
    fn hpr2(&mut self, n: usize, alpha: Complex<f64>, x: &[Complex<f64>], y: &[Complex<f64>], incx: usize, incy: usize) {
        let mut k = 0;
        for i in 0..n {
            for j in 0..=i {
                self[k] += alpha * x[i * incx] * y[j * incy].conj() + alpha.conj() * y[i * incy] * x[j * incx].conj();
                k += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_hpr2_complex_f32() {
        let mut ap = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let y = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        ap.as_mut_slice().hpr2(2, Complex::new(1.0, 0.0), x.as_slice(), y.as_slice(), 1, 1);
    }

    #[test]
    fn test_hpr2_complex_f64() {
        let mut ap = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let y = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        ap.as_mut_slice().hpr2(2, Complex::new(1.0, 0.0), x.as_slice(), y.as_slice(), 1, 1);
    }
}
