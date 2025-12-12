use num_complex::Complex;
use rayon::prelude::*;
use crate::level1::cabs::Cabs;

pub trait Iamax {
    type Output;
    fn iamax(&self, incx: usize) -> Self::Output;
}

impl Iamax for [f32] {
    type Output = usize;
    fn iamax(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .enumerate()
        .step_by(incx)
        .max_by(|(_, a), (_, b)|
                    a.partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Less)
                )
        .map(|(i, _)| i)
        .unwrap_or_default()
    }
}
impl Iamax for [f64] {
    type Output = usize;
    fn iamax(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .enumerate()
        .step_by(incx)
        .max_by(|(_, a), (_, b)|
                    a.partial_cmp(b)
                    .unwrap_or(std::cmp::Ordering::Less)
                )
        .map(|(i, _)| i)
        .unwrap_or_default()
    }
}
impl Iamax for [Complex<f32>] {
    type Output = usize;
    fn iamax(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .enumerate()
        .step_by(incx)
        .max_by(|&(_, c), &(_, d)|
                    c.cabs()
                    .partial_cmp(&d.cabs())
                    .unwrap_or(std::cmp::Ordering::Less)
                )
        .map(|(i, _)| i)
        .unwrap_or_default()
    }
}
impl Iamax for [Complex<f64>] {
    type Output = usize;
    fn iamax(&self, incx: usize) -> Self::Output {
        self
        .par_iter()
        .enumerate()
        .step_by(incx)
        .max_by(|&(_, c), &(_, d)|
                    c.cabs()
                    .partial_cmp(&d.cabs())
                    .unwrap_or(std::cmp::Ordering::Less)
                )
        .map(|(i, _)| i)
        .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_iamax() {
        let vector_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vector_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let vector_complex_f32: Vec<Complex<f32>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let vector_complex_f64: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];

        assert_eq!(vector_f32.as_slice().iamax(1), 4 as usize);
        assert_eq!(vector_f64.as_slice().iamax(3), 3 as usize);
        assert_eq!(vector_complex_f32.as_slice().iamax(1), 1 as usize);
        assert_eq!(vector_complex_f64.as_slice().iamax(1), 1 as usize);
    }
}