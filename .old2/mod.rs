#![allow(dead_code, unused)]

pub mod builder;
pub mod layer;
pub mod layer_builder;

use ndarray::prelude::*;

use crate::neural_net::layer::{activate, ActivationResult, Layer};

pub struct NN {
    features: Array2<f32>,
    target: Array1<f32>,
    layers: Vec<Layer>,
    learning_rate: f32,
    iteration: usize,
}

// Calculation methods
impl NN {
    /// Return the prediction of y based on x with current model parameters
    pub fn forward(&self, x: Array1<f32>) -> Array1<f32> {
        // y = input · weights + bias
        todo!()
    }
    fn calculate_mse(&self) -> Array2<f32> {
        todo!()
    }

    fn calculate_mse_gradient(&self) -> Array2<f32> {
        todo!()
    }
    fn update_params(&mut self, gradient: Array2<f32>) {
        todo!()
    }
}

// TRAINING LOOP
impl NN {
    pub fn run_training(&mut self) {
        println!("Starting training");
        (0..self.iteration).for_each(|_| self.train_cicle());
    }

    fn train_cicle(&mut self) {
        //let mse = self.calculate_mse();
        //dbg!(mse);
        let gradient = self.calculate_mse_gradient();
        //dbg!(gradient); // Aggiungi questo
        self.update_params(gradient);
        //dbg!(self.slope, self.bias); // E questo
    }
}
