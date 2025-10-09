#![allow(unused, non_snake_case)]

mod functions;
mod layer;
mod model;

use ndarray::{array, Array1, Array2, Axis};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

use crate::{
    functions::*,
    layer::{Activation, Layer},
    model::{Model, ModelBuilder},
};

const LEARNING_RATE: f32 = 0.02;
const EPOCHS: usize = 10000;

fn main() {
    // Input come vettori colonna (2, 1)
    let features = vec![
        array![[0.0_f32], [0.0]], // (2, 1)
        array![[1.0], [1.0]],     // (2, 1)
        array![[0.0], [1.0]],     // (2, 1)
        array![[1.0], [0.0]],     // (2, 1)
    ];
    let targets = vec![
        array![[0.0_f32]], // (1, 1)
        array![[0.0]],
        array![[1.0]],
        array![[1.0]],
    ];

    println!("Inizio training XOR...\n");
    ModelBuilder::default()
        .set_epoch(2000)
        .set_show_loss_every(100)
        .set_features(features)
        .set_targets(targets)
        .set_layers(vec![
            Layer::new("Hidden 1", 2, 3, Activation::Tanh),
            Layer::new("Hidden 2", 3, 2, Activation::Tanh),
            Layer::new("Output", 2, 1, Activation::Sigmoid),
        ])
        .build()
        // for the moment train is made for only this number of layers
        // also do not touch Output Activation function, do not change its act func or it's definition
        // see the
        .train();

    
}
