use num_complex::Complex;
use rayon::prelude::*;

/// Rank-1 update of a Hermitian matrix: A := alpha * x * conj(x)^T + A
pub trait Her {
    type Output;
    fn her(&mut self, n: usize, alpha: f32, x: &[Self::Output], incx: usize);
}

impl Her for [Complex<f32>] {
    type Output = Complex<f32>;
    fn her(&mut self, n: usize, alpha: f32, x: &[Complex<f32>], incx: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < n && col < n {
                    if row == col {
                        *a_elem += alpha * x[row * incx].norm_sqr();
                    } else if row < col {
                        *a_elem += alpha * x[row * incx] * x[col * incx].conj();
                    }
                }
            });
    }
}

impl Her for [Complex<f64>] {
    type Output = Complex<f64>;
    fn her(&mut self, n: usize, alpha: f32, x: &[Complex<f64>], incx: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < n && col < n {
                    if row == col {
                        *a_elem += alpha as f64 * x[row * incx].norm_sqr();
                    } else if row < col {
                        *a_elem += alpha as f64 * x[row * incx] * x[col * incx].conj();
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
    fn test_her_complex_f32() {
        let mut a = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                         Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        a.as_mut_slice().her(2, 1.0, x.as_slice(), 1);
    }

    #[test]
    fn test_her_complex_f64() {
        let mut a = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                         Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        a.as_mut_slice().her(2, 1.0, x.as_slice(), 1);
    }
}
