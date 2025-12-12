use num_complex::Complex;
use rayon::prelude::*;

pub trait Scal {
    type Output;
    fn scal(&mut self, a: Self::Output, incx: usize);
}

impl Scal for [f32] {
    type Output = f32;
    fn scal(&mut self, a: Self::Output, incx: usize) {
        self
        .par_iter_mut()
        .step_by(incx)
        .for_each(|x| *x *= a);
    }
}
impl Scal for [f64] {
    type Output = f64;
    fn scal(&mut self, a: Self::Output, incx: usize) {
        self
        .par_iter_mut()
        .step_by(incx)
        .for_each(|x| *x *= a);
    }
}
impl Scal for [Complex<f32>] {
    type Output = Complex<f32>;
    fn scal(&mut self, a: Self::Output, incx: usize) {
        self
        .par_iter_mut()
        .step_by(incx)
        .for_each(|x| *x *= a);
    }
}
impl Scal for [Complex<f64>] {
    type Output = Complex<f64>;
    fn scal(&mut self, a: Self::Output, incx: usize) {
        self
        .par_iter_mut()
        .step_by(incx)
        .for_each(|x| *x *= a);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_scal() {
        let mut vector_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut vector_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut vector_complex_f32: Vec<Complex<f32>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let mut vector_complex_f64: Vec<Complex<f64>> = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        
        vector_f32.as_mut_slice().scal(1.0, 1);
        assert_eq!(vector_f32, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        vector_f64.as_mut_slice().scal(2.0, 2);
        assert_eq!(vector_f64, vec![2.0, 2.0, 6.0, 4.0, 10.0]);

        vector_complex_f32.as_mut_slice().scal(Complex::new(1.0, 0.0), 1);
        assert_eq!(vector_complex_f32, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);

        vector_complex_f64.as_mut_slice().scal(Complex::new(1.0, 0.0), 1);
        assert_eq!(vector_complex_f64, vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
    }
}