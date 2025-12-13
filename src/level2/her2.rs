use num_complex::Complex;
use rayon::prelude::*;

/// Rank-2 update of a Hermitian matrix: A := alpha * x * conj(y)^T + conj(alpha) * y * conj(x)^T + A
pub trait Her2 {
    type Output;
    fn her2(&mut self, n: usize, alpha: Self::Output, x: &[Self::Output], y: &[Self::Output], incx: usize, incy: usize);
}

impl Her2 for [Complex<f32>] {
    type Output = Complex<f32>;
    fn her2(&mut self, n: usize, alpha: Complex<f32>, x: &[Complex<f32>], y: &[Complex<f32>], incx: usize, incy: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < n && col < n {
                    if row == col {
                        *a_elem += alpha * x[row * incx] * y[col * incy].conj() + alpha.conj() * y[row * incy] * x[col * incx].conj();
                    } else if row < col {
                        *a_elem += alpha * x[row * incx] * y[col * incy].conj() + alpha.conj() * y[row * incy] * x[col * incx].conj();
                    }
                }
            });
    }
}

impl Her2 for [Complex<f64>] {
    type Output = Complex<f64>;
    fn her2(&mut self, n: usize, alpha: Complex<f64>, x: &[Complex<f64>], y: &[Complex<f64>], incx: usize, incy: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < n && col < n {
                    if row == col {
                        *a_elem += alpha * x[row * incx] * y[col * incy].conj() + alpha.conj() * y[row * incy] * x[col * incx].conj();
                    } else if row < col {
                        *a_elem += alpha * x[row * incx] * y[col * incy].conj() + alpha.conj() * y[row * incy] * x[col * incx].conj();
                    }
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_her2_complex_f32() {
        let mut a = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                         Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let y = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        a.as_mut_slice().her2(2, Complex::new(1.0, 0.0), x.as_slice(), y.as_slice(), 1, 1);
    }

    #[test]
    fn test_her2_complex_f64() {
        let mut a = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                         Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        let y = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        a.as_mut_slice().her2(2, Complex::new(1.0, 0.0), x.as_slice(), y.as_slice(), 1, 1);
    }
}
