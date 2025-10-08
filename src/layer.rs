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
    forward_result: Option<(Array2<f32>,Array2<f32>)>,
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
    /// Ritorna (output_attivato, pre_attivazione)
    /// Questa è la versione giusta per backprop!
    pub(crate) fn forward(&mut self, input: &Array2<f32>) {
        // dbg!(&self.name, &input, &self.weights, &self.bias);

        let c = self.weights.t().dot(input) + &self.bias;
        let z = self.activate(&c);
        self.forward_result = Some((z, c));
    }

    pub fn backward(
        &mut self,
        dL_dZ_i: Array2<f32>,
    ) -> Array2<f32> {
        // dbg!(&self.name, &dL_dZ_i.shape(), &c_i.shape(), &z_im1.shape());
        // STEP 1: Applica la derivata dell'attivazione
        let (c_i, z_im1) = self.forward_result.as_ref().unwrap();
        let act_grad = self.activation_grad(c_i); // shape: stesso di c_i
        
        
        let delta_i = &dL_dZ_i * &act_grad; // element-wise, shape: (output_size, 1)
        // STEP 2: Gradiente dei pesi
        // dW = input^T · delta
        // z_im1: (input_size, 1)
        // delta_i: (output_size, 1)
        // Risultato: (input_size, output_size) - stesso shape di self.weights!
        // dbg!(&z_im1, delta_i.t());
        self.dL_dW = Some(z_im1.dot(&delta_i.t()));
        
        // STEP 3: Gradiente del bias
        self.dL_dB = Some(delta_i.clone());
        
        // STEP 4: Propaga l'errore al layer precedente
        // dL/dZ_(i-1) = W^T · delta_i
        
        // dbg!(&self.weights.shape(), &delta_i.shape());
        let prop = self.weights.dot(&delta_i);
        // dbg!(&prop.shape());
        // println!("\n\n");
        prop
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
            Activation::Sigmoid => array![[1.0]], // cuz it is already computed for the only activation that is output layer
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
            forward_result: None,
        }
    }
}
