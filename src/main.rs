#![allow(unused, non_snake_case)]

use ndarray::{array, Array1, Array2, Axis};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

const LEARNING_RATE: f32 = 0.02;
const EPOCHS: usize = 1000;

fn main() {
    // XOR INPUT
    let features = array![[0.0_f32, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
    // XOR outputs
    let targets = array![0.0_f32, 0.0, 1.0, 1.0];

    let input_size = 2;
    let hidden1_size = 3;
    let hidden2_size = 2;
    let output_size = 1;

    let mut h1 = Layer::new("Hidden 1", input_size, hidden1_size, Activation::Tanh);
    let mut h2 = Layer::new("Hidden 2", hidden1_size, hidden2_size, Activation::Tanh);
    let mut output = Layer::new("Output", hidden2_size, output_size, Activation::Sigmoid);

    println!("Inizio training XOR...\n");

    for epoch in 0..EPOCHS {
        let mut epoch_loss = 0.0;

        for (x, y_true) in features.axis_iter(Axis(0)).zip(targets.iter()) {
            let x = x.to_owned();

            // ============ FORWARD PASS ============
            // Ritorniamo sia pre-attivazione (c) che post-attivazione (z)
            let (z1, c1) = h1.forward(&x);
            let (z2, c2) = h2.forward(&z1);
            let (y_hat_arr, c_y) = output.forward(&z2);
            let y_hat = y_hat_arr[0];

            // Calcola loss
            let loss = loss(y_hat, *y_true);
            epoch_loss += loss;

            // ============ BACKWARD PASS ============

            // STEP 1: OUTPUT LAYER
            // Gradiente combinato di BCE e Sigmoid: y_hat - y_true
            let bce_sig_grad = bce_and_sigmoid_grad(y_hat, *y_true);

            // Gradiente dei pesi dell'output: dL/dW_output = z2^T · delta
            // z2 ha shape (hidden2_size,), delta è scalare
            // Risultato deve essere (hidden2_size, 1)
            let dL_dWy = z2.clone().insert_axis(Axis(1)) * bce_sig_grad;

            // Gradiente del bias dell'output
            let dL_dBy = bce_sig_grad;

            // STEP 2: PROPAGAZIONE A H2
            // dL/dz2 = W_output^T · delta_output
            // W_output ha shape (hidden2_size, 1), delta è scalare
            // Risultato: (hidden2_size,)
            let dL_dZ2 = output.weights.column(0).to_owned() * bce_sig_grad;

            // Applica la derivata di tanh: dL/dc2 = dL/dz2 · tanh'(c2)
            // tanh'(x) = 1 - tanh²(x)
            let tanh_grad_2 = c2.mapv(tanh_grad);
            let delta_h2 = &dL_dZ2 * &tanh_grad_2;

            // Gradiente dei pesi di H2: dL/dW_h2 = z1^T · delta_h2
            // z1 ha shape (hidden1_size,), delta_h2 ha shape (hidden2_size,)
            // Risultato: (hidden1_size, hidden2_size)
            let dL_dW2 = z1
                .clone()
                .insert_axis(Axis(1))
                .dot(&delta_h2.clone().insert_axis(Axis(0)));

            // Gradiente del bias di H2
            let dL_dB2 = delta_h2.clone();

            // STEP 3: PROPAGAZIONE A H1
            // dL/dz1 = W_h2 · delta_h2
            // W_h2 ha shape (hidden1_size, hidden2_size), delta_h2 ha shape (hidden2_size,)
            // Risultato: (hidden1_size,)
            let dL_dZ1 = h2.weights.dot(&delta_h2);

            // Applica la derivata di tanh: dL/dc1 = dL/dz1 · tanh'(c1)
            let tanh_grad_1 = c1.mapv(tanh_grad);
            let delta_h1 = &dL_dZ1 * &tanh_grad_1;

            // Gradiente dei pesi di H1: dL/dW_h1 = x^T · delta_h1
            // x ha shape (input_size,), delta_h1 ha shape (hidden1_size,)
            // Risultato: (input_size, hidden1_size)
            let dL_dW1 = x
                .clone()
                .insert_axis(Axis(1))
                .dot(&delta_h1.clone().insert_axis(Axis(0)));

            // Gradiente del bias di H1
            let dL_dB1 = delta_h1;

            // ============ UPDATE WEIGHTS ============
            output.weights = &output.weights - LEARNING_RATE * &dL_dWy;
            output.bias = &output.bias - LEARNING_RATE * array![dL_dBy];

            h2.weights = &h2.weights - LEARNING_RATE * &dL_dW2;
            h2.bias = &h2.bias - LEARNING_RATE * &dL_dB2;

            h1.weights = &h1.weights - LEARNING_RATE * &dL_dW1;
            h1.bias = &h1.bias - LEARNING_RATE * &dL_dB1;
        }

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss media = {:.4}", epoch, epoch_loss / 4.0);
        }
    }

    println!("\n=== RISULTATI FINALI ===");
    for (x, y_true) in features.axis_iter(Axis(0)).zip(targets.iter()) {
        let x = x.to_owned();
        let (z1, _) = h1.forward(&x);
        let (z2, _) = h2.forward(&z1);
        let (y_hat_arr, _) = output.forward(&z2);
        let y_hat = y_hat_arr[0];

        println!(
            "Input: [{:.0}, {:.0}] -> Predizione: {:.4}, Atteso: {:.0}",
            x[0], x[1], y_hat, y_true
        );
    }
}

struct Layer {
    name: String,
    weights: Array2<f32>,
    bias: Array1<f32>,
    act: Activation,
}

#[derive(Clone, Copy)]
enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
    Linear,
}

#[inline]
fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

#[inline]
fn tanh_grad(x: f32) -> f32 {
    1.0 - x.tanh().powi(2)
}

impl Layer {
    /// Ritorna (output_attivato, pre_attivazione)
    /// Questa è la versione giusta per backprop!
    fn forward(&self, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let c = input.dot(&self.weights) + &self.bias;
        let z = self.activate(&c);
        (z, c)
    }

    fn activate(&self, x: &Array1<f32>) -> Array1<f32> {
        match self.act {
            Activation::Tanh => x.mapv(f32::tanh),
            Activation::Sigmoid => x.mapv(sigmoid),
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Linear => x.clone(),
        }
    }

    fn new(
        name: &str,
        prev_layer_nodes: usize,
        current_layer_nodes: usize,
        act: Activation,
    ) -> Layer {
        Layer {
            name: name.into(),
            weights: Array2::<f32>::random((prev_layer_nodes, current_layer_nodes), StandardNormal),
            bias: Array1::<f32>::random(current_layer_nodes, StandardNormal),
            act,
        }
    }
}

fn loss(y_hat: f32, y_true: f32) -> f32 {
    // Binary Cross Entropy
    -y_true * y_hat.max(1e-7).ln() - (1.0 - y_true) * (1.0 - y_hat).max(1e-7).ln()
}

#[inline]
/// La derivata combinata di BCE e Sigmoid si semplifica in: y_hat - y_true
/// Questo è un trucco matematico molto utile!
fn bce_and_sigmoid_grad(y_hat: f32, y_true: f32) -> f32 {
    y_hat - y_true
}
