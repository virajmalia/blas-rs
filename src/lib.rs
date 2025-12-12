/*
Rust-native implementation for matrix operations
Design Goals:
1. Allow chaining of operations like matrix_obj1.add(matrix_obj2.mul(matrix_obj3))
2. Allow literal operations like matrix_obj1 + matrix_obj2 * matrix_obj3

Tasks:
1. Implement BLAS level 1 operations
2. Implement BLAS level 2 operations
3. Implement BLAS level 3 operations
4. Implement performance testing against netlib
*/
pub mod level1;
