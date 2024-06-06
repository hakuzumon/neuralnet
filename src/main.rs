use nalgebra::{DMatrix};
use crate::math::{matrix_average, slice_average};
use crate::network::Network;

mod math;
mod network;

fn main() {
    // train the meaning of XOR
    let training_data = vec![
        (column(&[0.0, 0.0]), column(&[0.0])),
        (column(&[1.0, 0.0]), column(&[1.0])),
        (column(&[0.0, 1.0]), column(&[1.0])),
        (column(&[1.0, 1.0]), column(&[0.0])),
    ];
    
    for _retries in 0..5 {
        let mut net = Network::new(vec![2, 4, 1]);
        for _ in 0..1000 {
            net.sgd(training_data.clone(), 1, 10, 20.0);
        }
    
        let test_data = training_data.clone();

        // println!("{}", net.feedforward(&column(&[0.0, 0.0]))[0]);
        // println!("{}", net.feedforward(&column(&[0.0, 1.0]))[0]);
        // println!("{}", net.feedforward(&column(&[1.0, 0.0]))[0]);
        // println!("{}", net.feedforward(&column(&[1.0, 1.0]))[0]);

        let avg_error = evaluate(&net, test_data);

        println!("accuracy {:.3}", 1.0 - avg_error);
    }
}

fn column(rows: &[f64]) -> DMatrix<f64> {
    DMatrix::from_row_slice(rows.len(), 1, rows)
}

fn evaluate(net: &Network, test_data: Vec<(DMatrix<f64>, DMatrix<f64>)>) -> f64 {
    let x: Vec<_> = test_data.iter()
        .map(|t| (net.feedforward(&t.0) - &t.1)
            .map(|a| a.abs()))
        .collect();
    
    let x1: Vec<_> = x.iter().map(matrix_average).collect();
    slice_average(&x1)
}
