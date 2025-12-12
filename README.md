# BLAS-RS: Native Rust Implementation of BLAS
`blas-rs` provides implementation of BLAS Level-1 APIs for Math and Scientific Computing.

## Features
1. Extends Rust's built-in Vector slices with BLAS methods.
2. Uses `num-complex` for complex number support.
3. Uses `rayon` for out-of-the-box parallelism.

## Usage

### Example 1
```rust
use blas_rs::level1::asum::Asum;

fn main() {
    let a_vec = vec![...];
    println!("Sum of all elements in a_vec is {}", a_vec.asum(1));
}
```

### Example 2
```rust
use blas_rs::level1::{axpy::Axpy, nrm2::Nrm2};

fn calc_euclidean_distance() -> f32 {
    // Cartesian coordinates of 2 points in 3D space
    let mut y: Vec<f32> = vec![1.0, 2.0, 3.0];
    let x: Vec<f32> = vec![4.0, 0.0, 8.0];

    // Distance = (-1)x + y, so alpha=-1.0
    x.axpy(-1.0, 1, &mut y, 1); // Compute y = y - x (in-place)
    y.nrm2(1) // Compute the Euclidean norm of the result
}
```

## Note
The BLAS standard was created with C and Fortran languages in mind, so when migrating to a Rust-based implementation, there are some design choices that require tweaking.

Currently implemented:
1. The first input array to the API is `self` of a `vector slice` to allow using the `.` operator on the rust vector slices directly.
2. Length parameters for all arrays are ommitted from the original BLAS APIs.
3. All other parameters follow the same order as the BLAS specification.

Some issues that need to be addressed:
1. BLAS spec implementations return a value only if the API outputs a scalar. A vector is outputted in-place. This leads to all `blas-rs` implemetations to return a `()` instead of a `slice of vector`. This maintains compliance with the BLAS spec, but blocks us from using method chains: `y.axpy(...).nrm2(...);`, and forces C-style calls: `y.axpy(...); y.nrm2(...);`.
2. There is no error-handling in the original BLAS spec APIs.

A possible fix could be to return something like `Result<T,BlasErr>` which can provide error-handling and also allow method chains through the `.` operator.