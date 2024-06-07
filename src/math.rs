use nalgebra::DMatrix;

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

/// Interpret the Matrix as vector and return the index of the largest value.
/// Fails if the argument is a multidimensional matrix.
///
/// Named after https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
pub fn argmax(m: &DMatrix<f64>) -> usize {
    assert!(m.ncols() == 1 || m.nrows() == 1);
    
    let mut max = f64::MIN;
    let mut max_index = 0;
    for (i, &a) in m.iter().enumerate() {
        if a > max {
            max = a;
            max_index = i;
        }
    }
    max_index
}