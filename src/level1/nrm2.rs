use num_complex::Complex;
use rayon::prelude::*;

pub trait Nrm2 {
    type Output;
    fn nrm2(&self, incx: usize) -> Self::Output;
}

impl Nrm2 for [f32] {
    type Output = f32;
    fn nrm2(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt()
    }
}
impl Nrm2 for [f64] {
    type Output = f64;
    fn nrm2(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
    }
}
impl Nrm2 for [Complex<f32>] {
    type Output = f32;
    fn nrm2(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .map(|x| x.re * x.re + x.im * x.im)
        .sum::<Self::Output>()
        .sqrt()
    }
}
impl Nrm2 for [Complex<f64>] {
    type Output = f64;
    fn nrm2(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .map(|x| x.re * x.re + x.im * x.im)
        .sum::<f64>()
        .sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_nrm2() {
        let vector_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vector_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vector_complex_f32: Vec<Complex<f32>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let vector_complex_f64: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        assert_eq!(vector_f32.as_slice().nrm2(1), 7.4161983f32);
        assert_eq!(vector_f64.as_slice().nrm2(2), 5.916079783099616f64);
        assert_eq!(vector_complex_f32.as_slice().nrm2(1), 5.477226f32);
        assert_eq!(vector_complex_f64.as_slice().nrm2(1), 5.477225575051661f64);
    }
}