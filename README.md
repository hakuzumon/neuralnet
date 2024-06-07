# Simple Neural Network in Rust

This Rust project is a port of the neural network example from [Chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html) of "Neural Networks and Deep Learning" by Michael Nielsen. Instead of using the MNIST dataset, this implementation uses the XOR function as the training example.

## Features

- Simple feedforward neural network
- Stochastic Gradient Descent for training
- Uses XOR function as the training example (not very useful, but lightweight)

## Usage

```sh
cargo run
```

## Example Output

```plaintext
Epoch 0: 2 / 4
Epoch 1: 2 / 4
Epoch 2: 2 / 4
...
Epoch 199: 4 / 4
```

The output varies somewhat from run to run, based on how well the network learns from the initial random weights and biases.

## Acknowledgments

- Original neural network example from [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen, licensed under [CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/deed.en).