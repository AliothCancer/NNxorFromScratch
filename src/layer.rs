use ndarray::{array, Array1, Array2};
use ndarray_rand::{rand_distr::StandardNormal, RandomExt};

use crate::{
    functions::{sigmoid, tanh, tanh_grad},
    LEARNING_RATE,
};

pub(crate) struct Layer {
    name: String,
    pub weights: Array2<f32>,
    pub bias: Array2<f32>,
    pub act: Activation,
    dL_dB: Option<Array2<f32>>,
    dL_dW: Option<Array2<f32>>,
}

#[derive(Clone, Copy)]
pub(crate) enum Activation {
    Tanh,
    Sigmoid,
    ReLU,
    Linear,
}

impl Layer {
    /// Ritorna (output attivato `z`, pre attivazione `c`)
    pub(crate) fn forward(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        let c = self.weights.t().dot(input) + &self.bias;
        let z = self.activate(&c);
        (z, c)
    }

    pub fn backward(
        &mut self,
        dL_dZ_i: &Array2<f32>,
        c_i: &Array2<f32>,
        z_im1: &Array2<f32>,
    ) -> Array2<f32> {
        let act_grad = self.activation_grad(c_i);
        let delta_i = dL_dZ_i * &act_grad;
        // dbg!(&delta_i);
        self.dL_dW = Some(z_im1.dot(&delta_i.t()));
        self.dL_dB = Some(delta_i.clone());
        self.weights.dot(&delta_i)
    }
    pub fn optimize(&mut self) {
        match (&self.dL_dW, &self.dL_dB) {
            (None, None) => panic!("Missing dL_dW and dL_dB, probably you didn't call backward first on this layer called {}", &self.name),
            (None, Some(_)) => panic!("Missing only dL_dW, should never happen"),
            (Some(_), None) => panic!("Missing only dL_dB, should never happen"),
            (Some(dW), Some(dB)) => {
                self.weights = &self.weights - LEARNING_RATE * dW;
                self.bias = &self.bias - LEARNING_RATE * dB;
            },
        }
    }
    pub fn activation_grad(&self, c_i: &Array2<f32>) -> Array2<f32> {
        match &self.act {
            Activation::Tanh => tanh_grad(c_i),
            Activation::Sigmoid => array![[1.0]], // cuz it is already computed for the ONLY Sigmoid activation that is the output layer, this derivative is already computed together with the bce one, the result really simplify calculation so that's why
            Activation::ReLU => todo!(),
            Activation::Linear => todo!(),
        }
    }
    pub(crate) fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        match self.act {
            Activation::Tanh => tanh(x),
            Activation::Sigmoid => sigmoid(x),
            Activation::ReLU => todo!(),
            Activation::Linear => x.to_owned(),
        }
    }

    pub(crate) fn new(
        name: &str,
        prev_layer_nodes: usize,
        current_layer_nodes: usize,
        act: Activation,
    ) -> Layer {
        Layer {
            name: name.into(),
            weights: Array2::<f32>::random((prev_layer_nodes, current_layer_nodes), StandardNormal),
            bias: Array2::<f32>::random((current_layer_nodes, 1), StandardNormal),
            act,
            dL_dB: None,
            dL_dW: None,
        }
    }
}
