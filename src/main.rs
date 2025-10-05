#![allow(unused)]

use ndarray::{array, Array1, Array2, ArrayView1, Axis};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

fn main() {
    // XOR INPUT
    let features = array![[0.0_f32, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
    // XOR outputs
    let targets = array![0.0_f32, 0.0, 1.0, 1.0];

    let input_size = 2;
    let hidden1_size = 3;
    let h1 = build_layer("Hidden 1", input_size, hidden1_size, Activation::Tanh);
    // h1.print();

    let hidden2_size = 2;
    let h2 = build_layer("Hidden 2", hidden1_size, hidden2_size, Activation::Tanh);
    // h2.print();

    let output_size = 1;
    let output = build_layer("Output", hidden2_size, output_size, Activation::Sigmoid);

    for (x, y_true) in features.axis_iter(Axis(0)).zip(targets.iter()) {
        let y_hat = forward(&x.to_owned(), (&h1, &h2, &output));
        let loss = loss(&y_hat, y_true);

        dbg!(&y_hat, &y_true);
        dbg!(&loss);
    }

    test_loss();
    // dbg!(first_el.t().dot(h1.weights));
    // let layers = vec![h1,h2, output];
    // forward(&first_el, layers);

    // println!("x = {}  ->  y = {}", x[1], y_pred);
}

struct Layer {
    name: String,
    weights: Array2<f32>,
    bias: Array1<f32>,
    act: Activation,
}
enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
    Linear,
}

fn linear_eval(input: &Array1<f32>, weights: &Array2<f32>, bias: &Array1<f32>) -> Array1<f32> {
    input.dot(weights) + bias
}
fn activate(act: &Activation, x: &Array1<f32>) -> Array1<f32> {
    match act {
        Activation::Tanh => x.mapv(f32::tanh),
        Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
        Activation::ReLU => x.mapv(|v| v.max(0.0)),
        Activation::Linear => x.clone(),
    }
}

impl Layer {
    fn print(&self) {
        println!(
            "\n\n\t ### {}\nWEIGHTS shape: {:?}\n\t{}\n",
            self.name,
            self.weights.shape(),
            self.weights
        );
        println!("BIAS shape: {:?}\n\t{}\n", self.bias.shape(), self.bias);
    }
}

fn build_layer(
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

fn forward(input: &Array1<f32>, layers: (&Layer, &Layer, &Layer)) -> f32 {
    let (h1, h2, output) = layers;

    let mut forward = activate(&h1.act, &(input.dot(&h1.weights) + &h1.bias));
    forward = activate(&h2.act, &(forward.dot(&h2.weights) + &h2.bias));
    forward = activate(&output.act, &(forward.dot(&output.weights) + &output.bias));
    forward[0]
}

fn loss(y_hat: &f32, y_true: &f32) -> f32 {
    match y_true {
        0.0 => -(1.0 - y_hat).ln(),
        1.0 => -y_hat.ln(),
        _ => unreachable!(),
    }
}


fn test_loss(){
    let y_hat = [0.1, 0.3, 0.7, 0.9];
    let y_true = [0.0, 0.0, 0.0, 0.0];
    let values = y_hat
        .iter()
        .zip(y_true.iter())
        .map(|(y_hat, y_true)| {
            let loss = loss(y_hat, y_true);
            let bce = -(1.0 - y_true - y_hat).abs().ln();

            (loss, bce)
        })
        .collect::<Vec<_>>();
    dbg!(values);

    let y_true = [
            1.0, 
            1.0, 
            1.0, 
            1.0
        ];
    let y_hat = [0.1, 0.3, 0.7, 0.9];
    let values = y_hat
        .iter()
        .zip(y_true.iter())
        .map(|(y_hat, y_true)| {
            let loss = loss(y_hat, y_true);
            let bce = -(1.0 - (y_true - y_hat).abs()).ln();

            (loss, bce)
        })
        .collect::<Vec<_>>();
    dbg!(values);
}
#[cfg(test)]
mod tests {
    use itertools::all;

    use super::*;

    #[test]
    fn test_loss_is_working() {
        let y_hat = [0.1, 0.3, 0.7, 0.9];
        let y_true = [
            1.0, 
            1.0, 
            1.0, 
            1.0
        ];

        let values = y_hat.iter().zip(y_true.iter()).all(|(y_hat, y_true)| {
            let loss = loss(y_hat, y_true);
            let bce = -(1.0 - (y_true - y_hat).abs()).ln();

            bce-0.001 < loss && loss < bce+0.001
        });

        assert!(values);
        // loss = -0.9.ln() ≈ 0.105
    }
}
