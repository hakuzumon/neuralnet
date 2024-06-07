use std::fmt::{Display, Formatter};
use nalgebra::{DMatrix};
use rand::{random, seq::SliceRandom};
use rand::thread_rng;
use crate::math::{argmax, sigmoid, sigmoid_prime};

/// Neural network based on http://neuralnetworksanddeeplearning.com/chap1.html
#[allow(dead_code)]
pub struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<DMatrix<f64>>,
    weights: Vec<DMatrix<f64>>
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Network {
        let mut biases: Vec<DMatrix<f64>> = Vec::new();
        let mut weights: Vec<DMatrix<f64>> = Vec::new();

        for &size in &sizes[1..] {
            let mut bias_layer = DMatrix::<f64>::zeros(size, 1);

            for x in 0..bias_layer.nrows() {
                bias_layer[(x, 0)] = random();
            }

            biases.push(bias_layer);
        }

        for (&x, &y) in sizes[..sizes.len() - 1].iter().zip(&sizes[1..]) {
            let mut weight_layer = DMatrix::<f64>::zeros(y, x);
            
            for i in 0..weight_layer.nrows() {
                for j in 0..weight_layer.ncols() {
                    weight_layer[(i, j)] = random();
                }
            }
            
            weights.push(weight_layer);
        }

        Network {
            num_layers: sizes.len(),
            sizes,
            biases,
            weights
        }
    }
    
    /// Return the output of the network if "a" is input.
    /// "a" is a n * 1 matrix.
    pub fn feedforward(&self, a: &DMatrix<f64>) -> DMatrix<f64> {
        let iter = self.biases.iter().zip(self.weights.iter());
        let mut x : DMatrix<f64> = a.clone();
        
        for (b, w) in iter {
            x = ((w * &x) + b).map(sigmoid);
        }
        
        x
    }
    
    // Stochastic Gradient Descent
    pub fn sgd(&mut self,
               mut training_data: Vec<(DMatrix<f64>, DMatrix<f64>)>,
               _epochs: usize,
               _mini_batch_size: usize,
               learning_rate: f64,
               test_data: Option<&Vec<(DMatrix<f64>, usize)>>
    ) {
        // todo implement epochs and mini_batches
        training_data.shuffle(&mut thread_rng());
        
        let mini_batch = training_data;
        self.update_mini_batch(&mini_batch, learning_rate);
        
        if let Some(td) = test_data {
            let test_result = self.evaluate(td);
            println!("Epoch: {} / {}", test_result, td.len());
        }
    }
    
    fn update_mini_batch(&mut self, mini_batch: &[(DMatrix<f64>, DMatrix<f64>)], learning_rate: f64) {
        let mut nabla_b: Vec<_> = self.biases.iter().map(zero_clone).collect();
        let mut nabla_w: Vec<_> = self.weights.iter().map(zero_clone).collect();

        for (input, output) in mini_batch.iter() {
            let (delta_nabla_b, delta_nabla_w) = self.backdrop(input, output);

            nabla_b = nabla_b.iter().zip(&delta_nabla_b).map(|(nb, dnb)| nb + dnb).collect();
            nabla_w = nabla_w.iter().zip(&delta_nabla_w).map(|(nw, dnw)| nw + dnw).collect();
        }

        let unit_learning_rate = learning_rate / mini_batch.len() as f64;
        self.weights = self.weights.iter().zip(&nabla_w).map(|(w, nw)| w - unit_learning_rate * nw).collect();
        self.biases = self.biases.iter().zip(&nabla_b).map(|(b, nb)| b - unit_learning_rate * nb).collect();
    }

    fn backdrop(&self, input: &DMatrix<f64>, output: &DMatrix<f64>) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        // TODO can all the cloning here be reduced?
        
        // why are we doing this again? not that it's expensive...
        let mut nabla_b: Vec<_> = self.biases.iter().map(zero_clone).collect();
        let mut nabla_w: Vec<_> = self.weights.iter().map(zero_clone).collect();
        
        // feedforward
        let mut activation = input.clone();
        let mut activations = vec![activation.clone()]; // list to store all activations, layer by layer
        let mut zs = Vec::<DMatrix<f64>>::new(); // list to store all the z vectors, layer by layer TODO what is a "z vector"?

        for (b, w) in self.biases.iter().zip(&self.weights) {
            let z = (w * &activation) + b;
            activation = z.map(sigmoid);
            activations.push(activation.clone());
            zs.push(z);
        }

        // backward pass
        let mut delta = cost_derivative(activations.last().unwrap(), output)
            .component_mul(&zs.last().unwrap().map(sigmoid_prime));

        let last_index = self.weights.len() - 1;
        nabla_b[last_index] = delta.clone();
        nabla_w[last_index] = &delta * activations[activations.len() - 2].transpose();

        // iterate from second-to-last layer to first layer 
        for lu in 2..self.num_layers {
            let l = lu as i32;
            let z = zs.get(from_last(&zs, -l)).unwrap();
            let sp = z.map(sigmoid_prime);
            let w = self.weights.get(from_last(&self.weights, -l + 1)).unwrap();
            delta = (w.transpose() * &delta).component_mul(&sp);
            let nabla_b_index = from_last(&nabla_b, -l); 
            let nabla_w_index = from_last(&nabla_w, -l); 
            nabla_b[nabla_b_index] = delta.clone();
            nabla_w[nabla_w_index] = &delta * activations[from_last(&activations, -l - 1)].transpose();
        }

        (nabla_b, nabla_w)
    }
    
    /// Return the number of test inputs for which the neural
    /// network outputs the correct result. Note that the neural
    /// network's output is assumed to be the index of whichever
    /// neuron in the final layer has the highest activation.
    fn evaluate(&self, test_data: &Vec<(DMatrix<f64>, usize)>) -> i32 {
        test_data.iter().map(|(input, output)|
            (argmax(&self.feedforward(input)) == *output) as i32
        ).sum()
    }
}

fn cost_derivative(output_activations: &DMatrix<f64>, output: &DMatrix<f64>) -> DMatrix<f64> {
    output_activations - output
}

// create a new matrix with same dimensions as input initialized as zeroes
fn zero_clone(input: &DMatrix<f64>) -> DMatrix<f64> {
    DMatrix::<f64>::zeros(input.nrows(), input.ncols())
}

impl Display for Network {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Network")?;

        // Print biases
        writeln!(f, "Biases:")?;
        writeln!(f, "1: (input layer)")?;
        for (i, bias_layer) in self.biases.iter().enumerate() {
            write!(f, "{}: {}", i + 2, bias_layer)?;
        }

        // Print weights
        writeln!(f, "Weights:")?;
        writeln!(f, "1: (input layer)")?;
        for (i, weight_layer) in self.weights.iter().enumerate() {
            write!(f, "{}: {}", i + 2, weight_layer)?;
        }

        Ok(())
    }
}

// Emulates python-like list indexing where index = -1 is the last index, -2 is the second-last, 
// and so on.
fn from_last<T>(slice: &[T], index: i32) -> usize {
    assert!(index < 0);
    let minus_index = index.abs() as usize; 
    let len = slice.len();
    len - minus_index
}