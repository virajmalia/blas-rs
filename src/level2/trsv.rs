use num_complex::Complex;

/// Solution of a linear system with a triangular matrix: A*x = b
pub trait Trsv {
    type Output;
    fn trsv(&self, n: usize, x: &mut [Self::Output], incx: usize);
}

impl Trsv for [f32] {
    type Output = f32;
    fn trsv(&self, n: usize, x: &mut [f32], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f32;
            for j in (i+1)..n {
                let a_idx = i * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            let a_ii = self[i * n + i];
            if a_ii != 0.0 {
                x[i * incx] = (x[i * incx] - sum) / a_ii;
            }
        }
    }
}

impl Trsv for [f64] {
    type Output = f64;
    fn trsv(&self, n: usize, x: &mut [f64], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f64;
            for j in (i+1)..n {
                let a_idx = i * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            let a_ii = self[i * n + i];
            if a_ii != 0.0 {
                x[i * incx] = (x[i * incx] - sum) / a_ii;
            }
        }
    }
}

impl Trsv for [Complex<f32>] {
    type Output = Complex<f32>;
    fn trsv(&self, n: usize, x: &mut [Complex<f32>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for j in (i+1)..n {
                let a_idx = i * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            let a_ii = self[i * n + i];
            if a_ii != Complex::new(0.0, 0.0) {
                x[i * incx] = (x[i * incx] - sum) / a_ii;
            }
        }
    }
}

impl Trsv for [Complex<f64>] {
    type Output = Complex<f64>;
    fn trsv(&self, n: usize, x: &mut [Complex<f64>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for j in (i+1)..n {
                let a_idx = i * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            let a_ii = self[i * n + i];
            if a_ii != Complex::new(0.0, 0.0) {
                x[i * incx] = (x[i * incx] - sum) / a_ii;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_trsv_f32() {
        let a_f32 = vec![2.0, 1.0, 0.0, 3.0];
        let mut x_f32 = vec![5.0, 6.0];
        
        a_f32.as_slice().trsv(2, &mut x_f32, 1);
    }

    #[test]
    fn test_trsv_f64() {
        let a_f64 = vec![2.0, 1.0, 0.0, 3.0];
        let mut x_f64 = vec![5.0, 6.0];
        
        a_f64.as_slice().trsv(2, &mut x_f64, 1);
    }

    #[test]
    fn test_trsv_complex_f32() {
        let a: Vec<Complex<f32>> = vec![
            Complex::new(1.0, 0.0), Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
        ];
        let mut x: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];
        
        a.as_slice().trsv(2, &mut x, 1);
    }
}
