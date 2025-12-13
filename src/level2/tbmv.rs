use num_complex::Complex;

/// Triangular band matrix-vector product: x = A * x
pub trait Tbmv {
    type Output;
    fn tbmv(&self, n: usize, k: usize, x: &mut [Self::Output], incx: usize);
}

impl Tbmv for [f32] {
    type Output = f32;
    fn tbmv(&self, n: usize, k: usize, x: &mut [f32], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f32;
            for j in (i+1)..n.min(i + k + 1) {
                let band_idx = k + j - i;
                let a_idx = band_idx * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            x[i * incx] = self[k * n + i] * x[i * incx] + sum;
        }
    }
}

impl Tbmv for [f64] {
    type Output = f64;
    fn tbmv(&self, n: usize, k: usize, x: &mut [f64], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = 0.0f64;
            for j in (i+1)..n.min(i + k + 1) {
                let band_idx = k + j - i;
                let a_idx = band_idx * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            x[i * incx] = self[k * n + i] * x[i * incx] + sum;
        }
    }
}

impl Tbmv for [Complex<f32>] {
    type Output = Complex<f32>;
    fn tbmv(&self, n: usize, k: usize, x: &mut [Complex<f32>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for j in (i+1)..n.min(i + k + 1) {
                let band_idx = k + j - i;
                let a_idx = band_idx * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            x[i * incx] = self[k * n + i] * x[i * incx] + sum;
        }
    }
}

impl Tbmv for [Complex<f64>] {
    type Output = Complex<f64>;
    fn tbmv(&self, n: usize, k: usize, x: &mut [Complex<f64>], incx: usize) {
        for i in (0..n).rev() {
            let mut sum = Complex::new(0.0, 0.0);
            for j in (i+1)..n.min(i + k + 1) {
                let band_idx = k + j - i;
                let a_idx = band_idx * n + j;
                if a_idx < self.len() {
                    sum += self[a_idx] * x[j * incx];
                }
            }
            x[i * incx] = self[k * n + i] * x[i * incx] + sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tbmv_f32() {
        let ab_f32 = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let mut x_f32 = vec![1.0, 2.0, 3.0];
        
        ab_f32.as_slice().tbmv(3, 1, &mut x_f32, 1);
    }

    #[test]
    fn test_tbmv_f64() {
        let ab_f64 = vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let mut x_f64 = vec![1.0, 2.0, 3.0];
        
        ab_f64.as_slice().tbmv(3, 1, &mut x_f64, 1);
    }
}
