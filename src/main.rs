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

const LEARNING_RATE: f32 = 0.02;
const EPOCHS: u32 = 60_000;


const H1: usize = 7;
const H2: usize = 8;


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
    let file = File::open("iris_encoded.csv").unwrap();
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

    ModelBuilder::default()
        .set_epoch(EPOCHS)
        .set_show_loss_every(EPOCHS / 10)
        .set_train_test_ratio(0.25)
        .set_features(features)
        .set_targets(targets)
        .set_layers(vec![
            Layer::new("Hidden 1", features_len, H1, Activation::Tanh),
            Layer::new("Hidden 2", H1, H2, Activation::Tanh),
            Layer::new("Output", H2, 3, Activation::Sigmoid),
        ])
        .build()
        .train()
        .metrics()
        .save_losses("losses.csv");
        // for the moment train is made for only this number of layers
        // also do not touch Output Activation function, do not change its act func or it's definition
        // see the `activation_grad` method for `Layer` struct
}
