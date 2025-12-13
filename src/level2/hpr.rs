use num_complex::Complex;

/// Rank-1 update of a Hermitian packed matrix: A := alpha * x * conj(x)^T + A
pub trait Hpr {
    type Output;
    fn hpr(&mut self, n: usize, alpha: f32, x: &[Self::Output], incx: usize);
}

impl Hpr for [Complex<f32>] {
    type Output = Complex<f32>;
    fn hpr(&mut self, n: usize, alpha: f32, x: &[Complex<f32>], incx: usize) {
        let mut k = 0;
        for i in 0..n {
            for j in 0..=i {
                if j == i {
                    self[k] += alpha * x[i * incx].norm_sqr();
                } else {
                    self[k] += alpha * x[i * incx] * x[j * incx].conj();
                }
                k += 1;
            }
        }
    }
}

impl Hpr for [Complex<f64>] {
    type Output = Complex<f64>;
    fn hpr(&mut self, n: usize, alpha: f32, x: &[Complex<f64>], incx: usize) {
        let alpha64 = alpha as f64;
        let mut k = 0;
        for i in 0..n {
            for j in 0..=i {
                if j == i {
                    self[k] += alpha64 * x[i * incx].norm_sqr();
                } else {
                    self[k] += alpha64 * x[i * incx] * x[j * incx].conj();
                }
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
    fn test_hpr_complex_f32() {
        let mut ap = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        ap.as_mut_slice().hpr(2, 1.0, x.as_slice(), 1);
    }

    #[test]
    fn test_hpr_complex_f64() {
        let mut ap = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        ap.as_mut_slice().hpr(2, 1.0, x.as_slice(), 1);
    }
}
