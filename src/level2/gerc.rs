use num_complex::Complex;
use rayon::prelude::*;

/// Rank-1 update of a conjugated general matrix: A := alpha * x * conj(y)^T + A
pub trait Gerc {
    type Output;
    fn gerc(&mut self, m: usize, n: usize, alpha: Self::Output, x: &[Self::Output], y: &[Self::Output], incx: usize, incy: usize);
}

impl Gerc for [Complex<f32>] {
    type Output = Complex<f32>;
    fn gerc(&mut self, m: usize, n: usize, alpha: Complex<f32>, x: &[Complex<f32>], y: &[Complex<f32>], incx: usize, incy: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < m && col < n {
                    *a_elem += alpha * x[row * incx] * y[col * incy].conj();
                }
            });
    }
}

impl Gerc for [Complex<f64>] {
    type Output = Complex<f64>;
    fn gerc(&mut self, m: usize, n: usize, alpha: Complex<f64>, x: &[Complex<f64>], y: &[Complex<f64>], incx: usize, incy: usize) {
        self.par_iter_mut()
            .enumerate()
            .for_each(|(i, a_elem)| {
                let row = i / n;
                let col = i % n;
                if row < m && col < n {
                    *a_elem += alpha * x[row * incx] * y[col * incy].conj();
                }
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_gerc_complex_f32() {
        let mut a = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                         Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let y = vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)];
        
        a.as_mut_slice().gerc(2, 2, Complex::new(1.0, 0.0), x.as_slice(), y.as_slice(), 1, 1);
        assert_eq!(a[0], Complex::new(1.0, -1.0));
        assert_eq!(a[1], Complex::new(2.0, -2.0));
    }

    #[test]
    fn test_gerc_complex_f64() {
        let mut a = vec![Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                         Complex::new(0.0, 0.0), Complex::new(0.0, 0.0)];
        let x = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0)];
        let y = vec![Complex::new(1.0, 1.0), Complex::new(2.0, 2.0)];
        
        a.as_mut_slice().gerc(2, 2, Complex::new(1.0, 0.0), x.as_slice(), y.as_slice(), 1, 1);
        assert_eq!(a[0], Complex::new(1.0, -1.0));
        assert_eq!(a[1], Complex::new(2.0, -2.0));
    }
}
