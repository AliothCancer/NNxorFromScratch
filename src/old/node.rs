use ndarray::Array1;
use rand::Rng;

pub struct Node {
    weights: Array1<f64>,
    bias: f64,
    learning_rate: f64,
}

impl Node {
    pub fn new(input_size: usize, learning_rate: f64) -> Node {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_shape_fn(input_size, |_| rng.gen_range(-1.0..1.0));
        let bias = rng.gen_range(-0.01..0.01);

        Node {
            weights,
            bias,
            learning_rate,
        }
    }

    pub fn predict(&self, input: Array1<f64>) -> f64 {
        let weighted_sum = self.weights.dot(&input) + self.bias;
        weighted_sum
    }

    // Funzione di attivazione sigmoid
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + f64::exp(-x))
    }

    // Derivata della funzione sigmoid
    pub fn sigmoid_derivative(x: f64) -> f64 {
        let sigmoid_x = Self::sigmoid(x);
        sigmoid_x * (1.0 - sigmoid_x)
    }
    pub fn train(&mut self, inputs: Array1<f64>, target: f64) {
        let weighted_sum = self.weights.dot(&inputs) + self.bias;
        let predicted = Self::sigmoid(weighted_sum);

        let error = target - predicted;
        let gradient = Self::sigmoid_derivative(weighted_sum) * error;

        for (i, weight) in self.weights.iter_mut().enumerate() {
            *weight += self.learning_rate * gradient * inputs[i];
        }
        self.bias += self.learning_rate * gradient;
    }
}

// Test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        use ndarray::Array1;
        let mut node = Node::new(3, 0.01);
        // Imposta pesi e bias noti per il test
        node.weights = Array1::from_vec(vec![0.5, -0.2, 0.3]);
        node.bias = 0.1;

        let inputs = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let predicted = node.predict(inputs);

        // Calcolo manuale dell'output atteso: (0.5*1.0) + (-0.2*2.0) + (0.3*3.0) + 0.1
        let expected_output = (0.5 * 1.0) + (-0.2 * 2.0) + (0.3 * 3.0) + 0.1;
        assert!((predicted - expected_output).abs() < 1e-6, "Expected {} but got {}", expected_output, predicted);
    }
}
