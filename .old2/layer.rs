
use ndarray::prelude::*;
use ndarray_rand::{rand_distr::{num_traits::Float, StandardNormal}, RandomExt};


// Represent all weights converging in one Node for every Node in the layer, so it is a matrix, (2 dim tensor), it has dimension n x m where n must be the number of inputs 
pub struct Layer{
    weights: Array2<f32>, // ha dimensione (n x m), n numero di
    biases: Array1<f32>, // vector
}

// Activation in mod.rs o separato
#[derive(Debug)]
pub enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
    Linear,
}

impl Activation {
    pub fn apply(&self, x: &Array1<f32>) -> Array1<f32> {
        match self {
            Activation::Tanh => x.mapv(|v| v.tanh()),
            Activation::Sigmoid => x.mapv(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::ReLU => x.mapv(|v| v.max(0.0)),
            Activation::Linear => x.clone(),
        }
    }
}
pub struct LayerShape{
    input_dim: usize,
    output_dim: usize 
}
pub enum LayerType{
    Input,
    Hidden,
    Activation(Activation),
    Output
}
impl LayerShape {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self { input_dim, output_dim }
    }
    fn get_dim(&self) -> (usize, usize){
        (self.input_dim, self.output_dim)
    }
}


impl Layer {
    pub fn new(shape: LayerShape) -> Self {
        let dim = shape.get_dim();
        let weights = Array2::random((shape.input_dim, shape.output_dim), StandardNormal);
        let biases = Array1::zeros(dim.1);
        
        Self { weights, biases}
    }
    
    pub fn forward(&self, inputs: &Array1<f32>) -> Array1<f32> {
        self.weights.dot(inputs) + &self.biases
    }
}

pub fn activate(layer: &Layer,input: ActivationResult) -> ActivationResult{
    todo!()
}

pub enum ActivationResult{
    Vector(Array1<f32>),
    Scalar(f32)
}
