use num_complex::Complex;

pub trait Trmv {
    type Output;
    fn trmv(&self, n: usize, x: &mut [Self::Output], incx: usize);
}

impl Trmv for [f32] {
    type Output = f32;
    fn trmv(&self, n: usize, x: &mut [f32], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f32;
            for j in (i+1)..n {
                let a_idx = i * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            x[i * incx] = self[i * n + i] * x[i * incx] + sum;
        }
    }
}

impl Trmv for [f64] {
    type Output = f64;
    fn trmv(&self, n: usize, x: &mut [f64], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f64;
            for j in (i+1)..n {
                let a_idx = i * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            x[i * incx] = self[i * n + i] * x[i * incx] + sum;
        }
    }
}

impl Trmv for [Complex<f32>] {
    type Output = Complex<f32>;
    fn trmv(&self, n: usize, x: &mut [Complex<f32>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for j in (i+1)..n {
                let a_idx = i * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            x[i * incx] = self[i * n + i] * x[i * incx] + sum;
        }
    }
}

impl Trmv for [Complex<f64>] {
    type Output = Complex<f64>;
    fn trmv(&self, n: usize, x: &mut [Complex<f64>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for j in (i+1)..n {
                let a_idx = i * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            x[i * incx] = self[i * n + i] * x[i * incx] + sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_trmv_f32() {
        let a_f32 = vec![1.0, 2.0, 0.0, 3.0];
        let mut x_f32 = vec![1.0, 2.0];
        
        a_f32.as_slice().trmv(2, &mut x_f32, 1);
        assert_eq!(x_f32[0], 13.0);
        assert_eq!(x_f32[1], 6.0);
    }

    #[test]
    fn test_trmv_f64() {
        let a_f64 = vec![1.0, 2.0, 0.0, 3.0];
        let mut x_f64 = vec![1.0, 2.0];
        
        a_f64.as_slice().trmv(2, &mut x_f64, 1);
        assert_eq!(x_f64[0], 13.0);
        assert_eq!(x_f64[1], 6.0);
    }

    #[test]
    fn test_trmv_complex_f32() {
        let a: Vec<Complex<f32>> = vec![
            Complex::new(1.0, 0.0), Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
        ];
        let mut x: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        a.as_slice().trmv(2, &mut x, 1);
    }
}
