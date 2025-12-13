/// Rank-2 update of a symmetric packed matrix: A := alpha * x * y^T + alpha * y * x^T + A
pub trait Spr2 {
    type Output;
    fn spr2(&mut self, n: usize, alpha: Self::Output, x: &[Self::Output], y: &[Self::Output], incx: usize, incy: usize);
}

impl Spr2 for [f32] {
    type Output = f32;
    fn spr2(&mut self, n: usize, alpha: f32, x: &[f32], y: &[f32], incx: usize, incy: usize) {
        let mut k = 0;
        for i in 0..n {
            for j in 0..=i {
                self[k] += alpha * (x[i * incx] * y[j * incy] + y[i * incy] * x[j * incx]);
                k += 1;
            }
        }
    }
}

impl Spr2 for [f64] {
    type Output = f64;
    fn spr2(&mut self, n: usize, alpha: f64, x: &[f64], y: &[f64], incx: usize, incy: usize) {
        let mut k = 0;
        for i in 0..n {
            for j in 0..=i {
                self[k] += alpha * (x[i * incx] * y[j * incy] + y[i * incy] * x[j * incx]);
                k += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spr2_f32() {
        let mut ap_f32 = vec![0.0, 0.0, 0.0];
        let x_f32 = vec![1.0, 2.0];
        let y_f32 = vec![1.0, 2.0];
        
        ap_f32.as_mut_slice().spr2(2, 1.0, x_f32.as_slice(), y_f32.as_slice(), 1, 1);
        assert_eq!(ap_f32, vec![2.0, 4.0, 8.0]);
    }

    #[test]
    fn test_spr2_f64() {
        let mut ap_f64 = vec![0.0, 0.0, 0.0];
        let x_f64 = vec![1.0, 2.0];
        let y_f64 = vec![1.0, 2.0];
        
        ap_f64.as_mut_slice().spr2(2, 1.0, x_f64.as_slice(), y_f64.as_slice(), 1, 1);
        assert_eq!(ap_f64, vec![2.0, 4.0, 8.0]);
    }
}
