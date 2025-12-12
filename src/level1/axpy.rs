use num_complex::Complex;
use rayon::prelude::*;

pub trait Axpy {
    type Output;
    fn axpy(&self, a: Self::Output, incx: usize, y: &mut Self, incy: usize);
}

impl Axpy for [f32] {
    type Output = f32;
    fn axpy(&self, a: f32, incx: usize, y: &mut Self, incy: usize) {
        self
        .par_iter()
        .step_by(incx)
        .zip(
            y
            .par_iter_mut()
            .step_by(incy)
        )
        .for_each(|(x, y)| {
            *y = a * *x + *y;
        })
    }
}
impl Axpy for [f64] {
    type Output = f64;
    fn axpy(&self, a: f64, incx: usize, y: &mut Self, incy: usize) {
        self
        .par_iter()
        .step_by(incx)
        .zip(
            y
            .par_iter_mut()
            .step_by(incy)
        )
        .for_each(|(x, y)| {
            *y = a * *x + *y;
        })
    }
}
impl Axpy for [Complex<f32>] {
    type Output = Complex<f32>;
    fn axpy(&self, a: Complex<f32>, incx: usize, y: &mut Self, incy: usize) {
        self
        .par_iter()
        .step_by(incx)
        .zip(
            y
            .par_iter_mut()
            .step_by(incy)
        )
        .for_each(|(x, y)| {
            *y = a * *x + *y;
        })
    }
}
impl Axpy for [Complex<f64>] {
    type Output = Complex<f64>;
    fn axpy(&self, a: Complex<f64>, incx: usize, y: &mut Self, incy: usize) {
        self
        .par_iter()
        .step_by(incx)
        .zip(
            y
            .par_iter_mut()
            .step_by(incy)
        )
        .for_each(|(x, y)| {
            *y = a * *x + *y;
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_axpy() {
        let x_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y_f32: Vec<f32> = vec![2.0; 5];
        let x_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y_f64 = vec![2.0; 5];
        let x_complex_f32: Vec<Complex<f32>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut y_complex_f32: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 2];
        let x_complex_f64: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut y_complex_f64: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 2];

        x_f32.as_slice().axpy(2.0, 1, &mut y_f32, 1);
        assert_eq!(y_f32, vec![4.0, 6.0, 8.0, 10.0, 12.0]);

        x_f64.as_slice().axpy(2.0, 2, &mut y_f64, 2);
        assert_eq!(y_f64, vec![4.0, 2.0, 8.0, 2.0, 12.0]);

        x_complex_f32.as_slice().axpy(Complex::new(1.0, 0.0), 1, &mut y_complex_f32, 1);
        assert_eq!(y_complex_f32, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        x_complex_f64.as_slice().axpy(Complex::new(1.0, 0.0), 1, &mut y_complex_f64, 1);
        assert_eq!(y_complex_f64, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    }
}