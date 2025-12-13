use num_complex::Complex;

/// Solution of a linear system with a triangular packed matrix: A*x = b
pub trait Tpsv {
    type Output;
    fn tpsv(&self, n: usize, x: &mut [Self::Output], incx: usize);
}

impl Tpsv for [f32] {
    type Output = f32;
    fn tpsv(&self, n: usize, x: &mut [f32], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f32;
            let mut k = i * (i + 1) / 2 + i + 1;
            for j in (i+1)..n {
                sum += self[k] * x[j * incx];
                k += 1;
            }
            let a_ii = self[i * (i + 1) / 2 + i];
            if a_ii != 0.0 {
                x[i * incx] = (x[i * incx] - sum) / a_ii;
            }
        }
    }
}

impl Tpsv for [f64] {
    type Output = f64;
    fn tpsv(&self, n: usize, x: &mut [f64], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f64;
            let mut k = i * (i + 1) / 2 + i + 1;
            for j in (i+1)..n {
                sum += self[k] * x[j * incx];
                k += 1;
            }
            let a_ii = self[i * (i + 1) / 2 + i];
            if a_ii != 0.0 {
                x[i * incx] = (x[i * incx] - sum) / a_ii;
            }
        }
    }
}

impl Tpsv for [Complex<f32>] {
    type Output = Complex<f32>;
    fn tpsv(&self, n: usize, x: &mut [Complex<f32>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            let mut k = i * (i + 1) / 2 + i + 1;
            for j in (i+1)..n {
                sum += self[k] * x[j * incx];
                k += 1;
            }
            let a_ii = self[i * (i + 1) / 2 + i];
            if a_ii != Complex::new(0.0, 0.0) {
                x[i * incx] = (x[i * incx] - sum) / a_ii;
            }
        }
    }
}

impl Tpsv for [Complex<f64>] {
    type Output = Complex<f64>;
    fn tpsv(&self, n: usize, x: &mut [Complex<f64>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            let mut k = i * (i + 1) / 2 + i + 1;
            for j in (i+1)..n {
                sum += self[k] * x[j * incx];
                k += 1;
            }
            let a_ii = self[i * (i + 1) / 2 + i];
            if a_ii != Complex::new(0.0, 0.0) {
                x[i * incx] = (x[i * incx] - sum) / a_ii;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tpsv_f32() {
        let ap_f32 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x_f32 = vec![1.0, 2.0, 3.0];
        
        ap_f32.as_slice().tpsv(3, &mut x_f32, 1);
    }

    #[test]
    fn test_tpsv_f64() {
        let ap_f64 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut x_f64 = vec![1.0, 2.0, 3.0];
        
        ap_f64.as_slice().tpsv(3, &mut x_f64, 1);
    }
}
