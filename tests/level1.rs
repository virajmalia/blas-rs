use blas_rs::level1::{axpy::Axpy, nrm2::Nrm2};

#[test]
fn calc_euclidean_distance() {
    // Cartesian coordinates of 2 points in 3D space
    let mut y: Vec<f32> = vec![1.0, 2.0, 3.0];
    let x: Vec<f32> = vec![4.0, 0.0, 8.0];

    // Distance = (-1)x + y, so alpha=-1.0
    x.axpy(-1.0, 1, &mut y, 1); // Compute y = y - x (in-place)
    let distance = y.nrm2(1); // Compute the Euclidean norm of the result
    
    let expected = 38.0_f32.sqrt();
    assert!((distance - expected).abs() < 1e-6);
}