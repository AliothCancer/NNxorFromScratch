use ndarray::{Array1, Array2};

use crate::node::Node;

pub struct NeuralNetwork {
    nodes: Vec<Node>,
    num_classes: usize,
}




impl NeuralNetwork {
    pub fn new(input_size: usize, num_classes: usize, learning_rate: f32) -> NeuralNetwork {
        let nodes = (0..num_classes)
            .map(|_| Node::new(input_size, learning_rate))
            .collect();
        NeuralNetwork {
            nodes,
            num_classes,
        }
    }

    pub fn predict(&self, input: Array1<f32>) -> usize {
        let predictions: Vec<f32> = self.nodes.iter()
            .map(|node| node.predict(&input))
            .collect();
        
        predictions.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn train(&mut self, inputs: Array2<f32>, targets: Array2<f32>, epochs: usize) {
        for _ in 0..epochs {
            let mut total_loss = 0.0;

            for (input, target) in inputs.outer_iter().zip(targets.outer_iter()) {
                for (i, node) in self.nodes.iter_mut().enumerate() {
                    let target_value = target[i];
                    let predicted = node.predict(input);
                    let error = target_value - predicted;
                    let gradient = Node::sigmoid_derivative(predicted) * error;
                    
                    node.train(&input, target_value);
                    total_loss += error.powi(2); // squared error
                }
            }

            println!("Epoch: {}, Loss: {}", _, total_loss / inputs.shape()[0] as f32);
        }
    }
}