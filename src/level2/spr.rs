/// Rank-1 update of a symmetric packed matrix: A := alpha * x * y^T + A
pub trait Spr {
    type Output;
    fn spr(&mut self, n: usize, alpha: Self::Output, x: &[Self::Output], incx: usize);
}

impl Spr for [f32] {
    type Output = f32;
    fn spr(&mut self, n: usize, alpha: f32, x: &[f32], incx: usize) {
        let mut k = 0;
        for i in 0..n {
            for j in 0..=i {
                self[k] += alpha * x[i * incx] * x[j * incx];
                k += 1;
            }
        }
    }
}

impl Spr for [f64] {
    type Output = f64;
    fn spr(&mut self, n: usize, alpha: f64, x: &[f64], incx: usize) {
        let mut k = 0;
        for i in 0..n {
            for j in 0..=i {
                self[k] += alpha * x[i * incx] * x[j * incx];
                k += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spr_f32() {
        let mut ap_f32 = vec![0.0, 0.0, 0.0];
        let x_f32 = vec![1.0, 2.0];
        
        ap_f32.as_mut_slice().spr(2, 1.0, x_f32.as_slice(), 1);
        assert_eq!(ap_f32, vec![1.0, 2.0, 4.0]);
    }

    #[test]
    fn test_spr_f64() {
        let mut ap_f64 = vec![0.0, 0.0, 0.0];
        let x_f64 = vec![1.0, 2.0];
        
        ap_f64.as_mut_slice().spr(2, 1.0, x_f64.as_slice(), 1);
        assert_eq!(ap_f64, vec![1.0, 2.0, 4.0]);
    }
}
