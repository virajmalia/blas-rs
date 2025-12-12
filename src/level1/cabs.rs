use num_complex::Complex;

pub trait Cabs {
    type Output;
    fn cabs(&self) -> Self::Output;
}

impl Cabs for Complex<f32> {
    type Output = f32;
    fn cabs(&self) -> Self::Output {
        self.re.abs() + self.im.abs()
    }
}
impl Cabs for Complex<f64> {
    type Output = f64;
    fn cabs(&self) -> Self::Output {
        self.re.abs() + self.im.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_cabs() {
        let vector_complex_f32: Complex<f32> = Complex::new(1.0, 2.0);
        let vector_complex_f64: Complex<f64> = Complex::new(1.0, 2.0);

        assert_eq!(vector_complex_f32.cabs(), 3.0);
        assert_eq!(vector_complex_f64.cabs(), 3.0);
    }
}