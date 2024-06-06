use nalgebra::DMatrix;

pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}

pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

pub fn matrix_average(m: &DMatrix<f64>) -> f64 {
    let sum: f64 = m.iter().sum();
    sum / m.len() as f64
}

pub fn slice_average(slice: &[f64]) -> f64 {
    let sum: f64 = slice.iter().sum();
    sum / slice.len() as f64
}