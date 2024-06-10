use nalgebra::DMatrix;
use rand::prelude::SliceRandom;
use rand::thread_rng;

use crate::network::Network;

mod math;
mod network;

fn main() {
    // train the meaning integer sum (up to 100)
    let mut training_data = generate_training_data(1);
    training_data.shuffle(&mut thread_rng());
    let tr2: Vec<_> = training_data.into_iter().take(100).collect();

    // output is the index of the neuron which is expected to activate the most, which we picked
    // to translate nicely to a boolean
    let test_data = vec![
        (column(&[normalize(2), normalize(2)]), column(&[normalize(4)])),
        (column(&[normalize(5), normalize(10)]), column(&[normalize(15)])),
        (column(&[normalize(75), normalize(1)]), column(&[normalize(76)])),
        (column(&[normalize(50), normalize(50)]), column(&[normalize(100)])),
        (column(&[normalize(99), normalize(99)]), column(&[normalize(198)])),
    ];
    
    let mut net = Network::new(vec![2, 8, 1]);
    
    net.sgd(tr2, 1000, 4, 25.0, Some(&test_data));
}

fn column(rows: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows.len(), 1, rows)
}

fn generate_training_data(repeats: usize) -> Vec<(DMatrix<f64>, DMatrix<f64>)> {
    let mut result: Vec<(DMatrix<f64>, DMatrix<f64>)> = Vec::new();

    for i in 0..100 {
        for j in 0..100 {
            let a = normalize(i);
            let b = normalize(j);
            let c = normalize(i + j);
            for _repeats in 0..repeats {
                result.push((column(&[a, b]), column(&[c])))
            }
        }
    }

    result
}

fn normalize(n: i32) -> f64 {
    n as f64 / 200.0
}

fn denormalize(n: f64) -> i32 {
    (n * 200.0) as i32
}