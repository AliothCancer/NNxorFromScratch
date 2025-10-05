use ndarray::prelude::*;

use crate::neural_net::{layer::{Layer, LayerType}, layer_builder::LayersBuilder, NN};

#[derive(Default)]
pub struct NNBuilder {
    x: Option<Array2<f32>>,
    y: Option<Array1<f32>>,
    layers: LayersBuilder,
    learning_rate: Option<f32>,
    iteration: Option<usize>,
}

impl NNBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn features(mut self, x: Array2<f32>) -> Self {
        self.x = Some(x);
        self
    }

    pub fn targets(mut self, y: Array1<f32>) -> Self {
        self.y = Some(y);
        self
    }

    pub fn add_layers(mut self, layers: Vec<(LayerType, usize)>) -> Self {
        todo!()
    }

    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn iteration(mut self, iteration: usize) -> Self {
        self.iteration = Some(iteration);
        self
    }

    pub fn build(self) -> Result<NN, String> {
        let layers = self.layers.build(); 
        if layers.is_empty() {
            return Err("At least one layer is required".to_string());
        }

        Ok(NN {
            features: self.x.ok_or("x is required")?,
            target: self.y.ok_or("y is required")?,
            layers,
            learning_rate: self.learning_rate.unwrap_or(0.01),
            iteration: self.iteration.unwrap_or(100),
        })
    }
}