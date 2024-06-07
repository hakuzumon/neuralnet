use nalgebra::DMatrix;

use crate::network::Network;

mod math;
mod network;

fn main() {
    // train the meaning of XOR -> boolean result (1st column = false, 2nd = true)
    let training_data = vec![
        (column(&[0.0, 0.0]), column(&[1.0, 0.0])),
        (column(&[1.0, 0.0]), column(&[0.0, 1.0])),
        (column(&[0.0, 1.0]), column(&[0.0, 1.0])),
        (column(&[1.0, 1.0]), column(&[1.0, 0.0])),
    ];
    
    // output is the index of the neuron which is expected to activate the most, which we picked
    // to translate nicely to a boolean
    let test_data = vec![
        (column(&[0.0, 0.0]), 0),
        (column(&[1.0, 0.0]), 1),
        (column(&[0.0, 1.0]), 1),
        (column(&[1.0, 1.0]), 0),
    ];
    
    let mut net = Network::new(vec![2, 4, 2]);
    
    net.sgd(training_data.clone(), 1, 10, 20.0, Some(&test_data));
}

fn column(rows: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows.len(), 1, rows)
}
