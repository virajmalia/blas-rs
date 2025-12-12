use num_complex::Complex;
use rayon::prelude::*;

pub trait Dot {
    type Output;
    fn dot(&self, incx: usize, y: &Self, incy: usize) -> Self::Output;
}

impl Dot for [f32] {
    type Output = f32;
    fn dot(&self, incx: usize, y: &Self, incy: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .zip(
            y
            .par_iter()
            .step_by(incy)
        )
        .map(|(x, y)| x * y)
        .sum()
    }
}
impl Dot for [f64] {
    type Output = f64;
    fn dot(&self, incx: usize, y: &Self, incy: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .zip(
            y
            .par_iter()
            .step_by(incy)
        )
        .map(|(x, y)| x * y)
        .sum()
    }
}
impl Dot for [Complex<f32>] {
    type Output = Complex<f32>;
    fn dot(&self, incx: usize, y: &Self, incy: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .zip(
            y
            .par_iter()
            .step_by(incy)
        )
        .map(|(x, y)| x * y)
        .sum()
    }
}
impl Dot for [Complex<f64>] {
    type Output = Complex<f64>;
    fn dot(&self, incx: usize, y: &Self, incy: usize) -> Self::Output {
        self
        .par_iter()
        .step_by(incx)
        .zip(
            y
            .par_iter()
            .step_by(incy)
        )
        .map(|(x, y)| x * y)
        .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_dot() {
        let x_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_f32: Vec<f32> = vec![2.0; 5];
        let x_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_f64 = vec![2.0; 5];
        let x_complex_f32: Vec<Complex<f32>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let y_complex_f32: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 2];
        let x_complex_f64: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let y_complex_f64: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 2];

        assert_eq!(x_f32.as_slice().dot(1, &y_f32, 1), 30.0);
        assert_eq!(x_f64.as_slice().dot(2, &y_f64, 2), 18.0);
        assert_eq!(x_complex_f32.as_slice().dot(1, &y_complex_f32, 1), Complex::new(0.0, 0.0));
        assert_eq!(x_complex_f64.as_slice().dot(1, &y_complex_f64, 1), Complex::new(0.0, 0.0));
    }
}