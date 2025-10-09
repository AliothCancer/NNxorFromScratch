#![allow(unused, non_snake_case)]

mod functions;
mod layer;
mod model;
mod metrics;

use std::fs::File;

use csv::ReaderBuilder;
use itertools::Itertools;
use ndarray::{array, s, Array1, Array2, Axis};
use ndarray_csv::Array2Reader;
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};
use serde::{Deserialize, Serialize};

use crate::{
    functions::*,
    layer::{Activation, Layer},
    model::{Model, ModelBuilder},
};

const LEARNING_RATE: f32 = 0.002;
const EPOCHS: u32 = 60_000;

const INPUT: usize = 3;
const H1: usize = 3;
const H2: usize = 2;
const OUTPUT: usize = 4;

#[derive(Deserialize, Debug)]
struct IrisRecord {
    sepal_length: f32,
    sepal_width: f32,
    petal_length: f32,
    petal_width: f32,
    target1: f32,
    target2: f32,
    target3: f32,
}

fn main() {
    let file = File::open("/home/giulio/Scrivania/NeuralFromScratch/iris_encoded.csv").unwrap();
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .delimiter(b',')
        .double_quote(false)
        .from_reader(file);
    let dataset = reader.deserialize_array2::<f32>((150, 7)).unwrap();

    let features_len = 4;
    let features = dataset
        .slice(s![.., 0..features_len]).insert_axis(Axis(2)).to_owned();
    dbg!(features.shape());
    let targets = dataset.slice(s![.., features_len..]).insert_axis(Axis(2)).to_owned();
    dbg!(&targets.shape());
    // Input come vettori colonna (2, 1)

    ModelBuilder::default()
        .set_epoch(EPOCHS)
        .set_show_loss_every(EPOCHS / 10)
        .set_features(features)
        .set_targets(targets)
        .set_layers(vec![
            Layer::new("Hidden 1", features_len, 3, Activation::Tanh),
            Layer::new("Hidden 2", 3, 6, Activation::Tanh),
            Layer::new("Output", 6, 3, Activation::Sigmoid),
        ])
        .build()
        .train()
        .metrics()
        .save_losses("losses.csv");
        // for the moment train is made for only this number of layers
        // also do not touch Output Activation function, do not change its act func or it's definition
        // see the `activation_grad` method for `Layer` struct
}
