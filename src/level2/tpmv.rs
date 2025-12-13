use num_complex::Complex;

/// Triangular packed matrix-vector product: x = A * x
pub trait Tpmv {
    type Output;
    fn tpmv(&self, n: usize, x: &mut [Self::Output], incx: usize);
}

impl Tpmv for [f32] {
    type Output = f32;
    fn tpmv(&self, n: usize, x: &mut [f32], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f32;
            let mut k = i * (i + 1) / 2 + i + 1;
            for j in (i+1)..n {
                sum += self[k] * x[j * incx];
                k += 1;
            }
            x[i * incx] = self[i * (i + 1) / 2 + i] * x[i * incx] + sum;
        }
    }
}

impl Tpmv for [f64] {
    type Output = f64;
    fn tpmv(&self, n: usize, x: &mut [f64], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f64;
            let mut k = i * (i + 1) / 2 + i + 1;
            for j in (i+1)..n {
                sum += self[k] * x[j * incx];
                k += 1;
            }
            x[i * incx] = self[i * (i + 1) / 2 + i] * x[i * incx] + sum;
        }
    }
}

impl Tpmv for [Complex<f32>] {
    type Output = Complex<f32>;
    fn tpmv(&self, n: usize, x: &mut [Complex<f32>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            let mut k = i * (i + 1) / 2 + i + 1;
            for j in (i+1)..n {
                sum += self[k] * x[j * incx];
                k += 1;
            }
            x[i * incx] = self[i * (i + 1) / 2 + i] * x[i * incx] + sum;
        }
    }
}

impl Tpmv for [Complex<f64>] {
    type Output = Complex<f64>;
    fn tpmv(&self, n: usize, x: &mut [Complex<f64>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            let mut k = i * (i + 1) / 2 + i + 1;
            for j in (i+1)..n {
                sum += self[k] * x[j * incx];
                k += 1;
            }
            x[i * incx] = self[i * (i + 1) / 2 + i] * x[i * incx] + sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpmv_f32() {
        let ap_f32 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x_f32 = vec![1.0, 2.0, 3.0];
        
        ap_f32.as_slice().tpmv(3, &mut x_f32, 1);
    }

    #[test]
    fn test_tpmv_f64() {
        let ap_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x_f64 = vec![1.0, 2.0, 3.0];
        
        ap_f64.as_slice().tpmv(3, &mut x_f64, 1);
    }
}
