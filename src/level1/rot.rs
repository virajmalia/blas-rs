use num_complex::Complex;
use rayon::prelude::*;

pub trait Rot {
    type Output;
    fn rot(&mut self, incx: usize, y: &mut Self, incy: usize, c: Self::Output, s: Self::Output);
}

impl Rot for [f32] {
    type Output = f32;
    fn rot(&mut self, incx: usize, y: &mut Self, incy: usize, c: f32, s: f32) {
        self
        .par_iter_mut()
        .step_by(incx)
        .zip(
            y
            .par_iter_mut()
            .step_by(incy)
        )
        .for_each(|(x, y)| {
            let temp = c * *x + s * *y;
            *y = c * *y - s * *x;
            *x = temp;
        })
    }
}
impl Rot for [f64] {
    type Output = f64;
    fn rot(&mut self, incx: usize, y: &mut Self, incy: usize, c: f64, s: f64) {
        self
        .par_iter_mut()
        .step_by(incx)
        .zip(
            y
            .par_iter_mut()
            .step_by(incy)
        )
        .for_each(|(x, y)| {
            let temp = c * *x + s * *y;
            *y = c * *y - s * *x;
            *x = temp;
        })
    }
}
impl Rot for [Complex<f32>] {
    type Output = Complex<f32>;
    fn rot(&mut self, incx: usize, y: &mut Self, incy: usize, c: Complex<f32>, s: Complex<f32>) {
        self
        .par_iter_mut()
        .step_by(incx)
        .zip(
            y
            .par_iter_mut()
            .step_by(incy)
        )
        .for_each(|(x, y)| {
            let temp = c * *x + s * *y;
            *y = c * *y - s * *x;
            *x = temp;
        })
    }
}
impl Rot for [Complex<f64>] {
    type Output = Complex<f64>;
    fn rot(&mut self, incx: usize, y: &mut Self, incy: usize, c: Complex<f64>, s: Complex<f64>) {
        self
        .par_iter_mut()
        .step_by(incx)
        .zip(
            y
            .par_iter_mut()
            .step_by(incy)
        )
        .for_each(|(x, y)| {
            let temp = c * *x + s * *y;
            *y = c * *y - s * *x;
            *x = temp;
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_rot() {
        let mut x_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y_f32: Vec<f32> = vec![2.0; 5];
        let mut x_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut y_f64 = vec![2.0; 5];
        let mut x_complex_f32: Vec<Complex<f32>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut y_complex_f32: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); 2];
        let mut x_complex_f64: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut y_complex_f64: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); 2];

        x_f32.as_mut_slice().rot(1, &mut y_f32, 1, 2.0, 2.0);
        assert_eq!(x_f32, vec![6.0, 8.0, 10.0, 12.0, 14.0]);
        assert_eq!(y_f32, vec![2.0, 0.0, -2.0, -4.0, -6.0]);

        x_f64.as_mut_slice().rot(2, &mut y_f64, 2, 2.0, 2.0);
        assert_eq!(x_f64, vec![6.0, 2.0, 10.0, 4.0, 14.0]);
        assert_eq!(y_f64, vec![2.0, 2.0, -2.0, 2.0, -6.0]);

        x_complex_f32.as_mut_slice().rot(1, &mut y_complex_f32, 1, Complex::new(1.0, 0.0), Complex::new(0.0, 1.0));
        assert_eq!(x_complex_f32, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(y_complex_f32, vec![Complex::new(2.0, -1.0), Complex::new(4.0, -3.0)]);

        x_complex_f64.as_mut_slice().rot(1, &mut y_complex_f64, 1, Complex::new(1.0, 0.0), Complex::new(0.0, 1.0));
        assert_eq!(x_complex_f64, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        assert_eq!(y_complex_f64, vec![Complex::new(2.0, -1.0), Complex::new(4.0, -3.0)]);
    }
}