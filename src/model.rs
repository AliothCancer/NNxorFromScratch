use ndarray::{array, Array2};

use crate::{functions::bce_and_sigmoid_grad, layer::Layer};

#[derive(Default)]
pub struct Model {
    features: Vec<Array2<f32>>,
    targets: Vec<Array2<f32>>,
    layers: Vec<Layer>,
    epoch: u32,
}

impl Model {
    pub fn new() -> Self {
        Model::default()
    }
    pub fn add_layers(&mut self, layer: Layer) {
        self.layers.push(layer);
    }
    pub fn set_target(&mut self, target: Vec<Array2<f32>>) {
        self.targets = target
    }
    pub fn set_features(&mut self, features: Vec<Array2<f32>>) {
        self.features = features
    }
    
    pub fn train(&self) {
        for epoch in (0..self.epoch) {
            let mut epoch_loss = array![[0.0]];
            for (x, y_true) in self.features.iter().zip(self.targets.iter()) {
                                // dbg!(&x.shape(),&y_true.shape());
                let x = x.to_owned();
                
                for layer in self.layers.iter_mut(){
                    let (z,c) = layer.forward(&x);
                }
                let (z1, c1) = h1.forward(&x);
                let (z2, c2) = h2.forward(&z1);
                let (y_hat, c_y) = output.forward(&z2);
                
                // Calcola loss
                let loss = loss(&y_hat, y_true);
                // dbg!(&loss);
                epoch_loss = epoch_loss + loss;
                
                // dbg!(&propagation);
                let mut propagation = array![[0.0]];
                    propagation = output.backward(propagation, c_y, z2);
                    propagation = h2.backward(propagation, c2, z1);
                    propagation = h1.backward(propagation, c1, x);
    
                    output.optimize();
                    h2.optimize();
                    h1.optimize();

                }
        }

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss media = {:.4}", epoch, epoch_loss / 4.0);
        }
        

        println!("\n=== RISULTATI FINALI ===");
        
    }
    
    pub fn set_epoch(&mut self, epoch: u32) {
        self.epoch = epoch;
    }
}
