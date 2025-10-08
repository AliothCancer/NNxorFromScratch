#![allow(unused, non_snake_case)]

mod functions;
mod layer;
mod model;

use ndarray::{array, Array1, Array2, Axis};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

use crate::{
    functions::*,
    layer::{Activation, Layer},
};

const LEARNING_RATE: f32 = 0.02;
const EPOCHS: usize = 10000;

fn main() {
    // Input come vettori colonna (2, 1)
    let features = array![
        [[0.0_f32],[0.0]], // (2, 1)
        [[1.0],[1.0]],     // (2, 1)
        [[0.0],[1.0]],     // (2, 1)
        [[1.0],[0.0]]      // (2, 1)
    ];
    let targets = array![
        [[1.0_f32]], // (1, 1)
        [[0.0]],
        [[1.0]],
        [[1.0]]
    ];

    let mut h1 = Layer::new("Hidden 1", 2, 3, Activation::Tanh);
    let mut h2 = Layer::new("Hidden 2", 3, 2, Activation::Tanh);
    let mut output = Layer::new("Output", 2, 1, Activation::Sigmoid);

    println!("Inizio training XOR...\n");
    // dbg!(
    //     array![[1,2]].t().dot(&array![[3, 4]])
    // );
    
}
