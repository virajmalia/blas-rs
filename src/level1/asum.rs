use num_complex::Complex;
use rayon::prelude::*;
use crate::level1::cabs::Cabs;

pub trait Asum {
    type Output;
    fn asum(&self, incx: usize) -> Self::Output;
}

impl Asum for [f32] {
    type Output = f32;
    fn asum(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .map(|x| x.abs())
        .sum()
    }
}
impl Asum for [f64] {
    type Output = f64;
    fn asum(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .map(|x| x.abs())
        .sum()
    }
}
impl Asum for [Complex<f32>] {
    type Output = f32;
    fn asum(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .map(|c| c.cabs())
        .sum()
    }
}
impl Asum for [Complex<f64>] {
    type Output = f64;
    fn asum(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .map(|c| c.cabs())
        .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_asum() {
        let vector_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vector_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vector_complex_f32: Vec<Complex<f32>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let vector_complex_f64: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        assert_eq!(vector_f32.as_slice().asum(1), 15.0);
        assert_eq!(vector_f64.as_slice().asum(2), 9.0);
        assert_eq!(vector_complex_f32.as_slice().asum(1), 10.0);
        assert_eq!(vector_complex_f64.as_slice().asum(1), 10.0);
    }
}