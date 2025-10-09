use std::default;

use ndarray::{array, Array2};

use crate::{
    functions::{bce_and_sigmoid_grad, loss},
    layer::Layer,
};

#[derive(Default)]
pub struct Model {
    features: Vec<Array2<f32>>,
    targets: Vec<Array2<f32>>,
    layers: Vec<Layer>,
    epoch: u32,
    show_loss_every: u32,
}

impl Model {
    pub fn train(&mut self) {
        for epoch in (0..=self.epoch) {
            let mut epoch_loss = array![[0.0]];
            for (x, y_true) in self.features.iter().zip(self.targets.iter()) {
                // dbg!(&x.shape(),&y_true.shape());
                let x = x.to_owned();

                let mut layers = self.layers.as_mut_slice();
                let (h1, rest) = layers.split_first_mut().unwrap();
                let (h2, rest) = rest.split_first_mut().unwrap();
                let output = &mut rest[0];

                let (z1, c1) = h1.forward(&x);
                // dbg!(&z1,&c1);
                let (z2, c2) = h2.forward(&z1);
                let (y_hat, c_y) = output.forward(&z2);

                // Calcola loss
                let loss = loss(&y_hat, y_true);
                // dbg!(&loss);
                epoch_loss = epoch_loss + loss;

                // dbg!(&propagation);
                let propagation = bce_and_sigmoid_grad(&y_hat, y_true);
                let propagation = output.backward(&propagation, &c_y, &z2);
                let propagation = h2.backward(&propagation, &c2, &z1);
                let propagation = h1.backward(&propagation, &c1, &x);

                output.optimize();
                h2.optimize();
                h1.optimize();
            }
            if epoch % self.show_loss_every == 0 {
                println!("Epoch {}: Loss media = {:.4}", epoch, epoch_loss[(0,0)] / 4.0);
            }
        }

        println!("\n=== RISULTATI FINALI ===");
        for (x, y_true) in self.features.iter().zip(self.targets.iter()) {
            let x = x.to_owned();
            let mut layers = self.layers.as_mut_slice();
            let (h1, rest) = layers.split_first_mut().unwrap();
            let (h2, rest) = rest.split_first_mut().unwrap();
            let output = &mut rest[0];

            let (z1, c1) = h1.forward(&x);
            let (z2, c2) = h2.forward(&z1);
            let (y_hat, c_y) = output.forward(&z2);

            println!(
                "Input: [{}, {}]-> Predizione: {:?}, Atteso: {:?}", x[(0,0)],x[(1,0)], y_hat[(0,0)], y_true[(0,0)]
            );
        }
    }
}
#[derive(Default)]
pub struct ModelBuilder {
    features: Option<Vec<Array2<f32>>>,
    targets: Option<Vec<Array2<f32>>>,
    layers: Option<Vec<Layer>>,
    epoch: Option<u32>,
    show_loss_every: Option<u32>,
}
impl ModelBuilder {
    pub fn build(self) -> Model {
        Model {
            features: self.features.unwrap(),
            targets: self.targets.unwrap(),
            layers: self.layers.unwrap(),
            epoch: self.epoch.unwrap(),
            show_loss_every: self.show_loss_every.unwrap_or(0),
        }
    }
    pub fn set_layers(mut self, layers: Vec<Layer>) -> Self {
        Self {
            layers: Some(layers),
            ..self
        }
    }
    pub fn set_epoch(mut self, epoch: u32) -> Self {
        Self {
            epoch: Some(epoch),
            ..self
        }
    }
    pub fn set_targets(mut self, targets: Vec<Array2<f32>>) -> Self {
        Self {
            targets: Some(targets),
            ..self
        }
    }
    pub fn set_features(mut self, features: Vec<Array2<f32>>) -> Self {
        Self {
            features: Some(features),
            ..self
        }
    }

    pub fn set_show_loss_every(mut self, show_loss_every: u32) -> Self {
        Self {
            show_loss_every: Some(show_loss_every),
            ..self
        }
    }
}
