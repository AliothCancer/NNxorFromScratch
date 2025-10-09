use std::{collections::HashMap, default, fs::File};

use csv::WriterBuilder;
use ndarray::{array, Array2, Array3, Axis, Order};
use ndarray_csv::Array2Writer;
use serde::Serialize;

use crate::{
    functions::{bce_and_sigmoid_grad, loss},
    layer::Layer,
    metrics::ClassMetrics,
};

#[derive(Default)]
pub struct Model {
    features: Array3<f32>,
    targets: Array3<f32>,
    layers: Vec<Layer>,
    epoch: u32,
    show_loss_every: u32,
    losses: Vec<Array2<f32>>,
}

impl Model {
    pub fn train(mut self) -> Self {
        let feat_len = self.features.len_of(Axis(0));
        // dbg!(self.targets.outer_iter().take(1).last().unwrap());
        for epoch in (0..=self.epoch) {
            let mut epoch_loss = array![[0.0]];
            for (x, y_true) in self.features.outer_iter().zip(self.targets.outer_iter()) {
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
            let mean_epoch_loss = epoch_loss / (feat_len as f32);
            self.losses.push(mean_epoch_loss.clone());
            // if epoch % self.show_loss_every == 0 {
            //     println!(
            //         "Epoch {}: Loss media = {:.4}",
            //         epoch,
            //         mean_epoch_loss
            //     );
            // }
        }

        self
    }

    pub(crate) fn save_losses(&self, path: &str) {
        let mut wtr = WriterBuilder::new()
            .delimiter(b',')
            .has_headers(false)
            .from_path(path)
            .unwrap();

        // dbg!(&self.losses.len());
        for loss in self.losses.iter() {
            wtr.serialize_array2(&loss.t().to_owned()).unwrap();
        }
    }

    pub fn metrics(mut self) -> Self {
        println!("\n=== CALCOLO METRICHE ===");

        let class_names = ["Iris-setosa", "Iris-virginica", "Iris-versicolor"];
        let mut metrics: HashMap<usize, ClassMetrics> = HashMap::new();
        for i in 0..3 {
            metrics.insert(i, ClassMetrics::default());
        }

        let mut correct = 0;
        let mut total = 0;

        for (x, y_true) in self
            .features
            .outer_iter()
            .zip(self.targets.outer_iter())
        {
            let x = x.to_owned();
            let mut layers = self.layers.as_mut_slice();
            let (h1, rest) = layers.split_first_mut().unwrap();
            let (h2, rest) = rest.split_first_mut().unwrap();
            let output = &mut rest[0];

            let (z1, _) = h1.forward(&x);
            let (z2, _) = h2.forward(&z1);
            let (y_hat, _) = output.forward(&z2);

            // Trova la classe predetta (argmax)
            let mut predicted_class = 0;
            let mut max_val = y_hat[[0, 0]];
            for i in 1..3 {
                if y_hat[[i, 0]] > max_val {
                    max_val = y_hat[[i, 0]];
                    predicted_class = i;
                }
            }

            // Trova la classe vera (argmax)
            let mut true_class = 0;
            let mut max_val = y_true[[0, 0]];
            for i in 1..3 {
                if y_true[[i, 0]] > max_val {
                    max_val = y_true[[i, 0]];
                    true_class = i;
                }
            }

            // Aggiorna contatori
            if predicted_class == true_class {
                correct += 1;
            }
            total += 1;

            // Aggiorna metriche per ogni classe
            for class_idx in 0..3 {
                let metric = metrics.get_mut(&class_idx).unwrap();

                if true_class == class_idx && predicted_class == class_idx {
                    metric.true_positives += 1;
                } else if true_class != class_idx && predicted_class == class_idx {
                    metric.false_positives += 1;
                } else if true_class == class_idx && predicted_class != class_idx {
                    metric.false_negatives += 1;
                } else {
                    metric.true_negatives += 1;
                }
            }
        }
        // Stampa risultati
        let accuracy = correct as f64 / total as f64;
        println!("\n=== METRICHE GLOBALI ===");
        println!("Accuracy: {:.4} ({}/{})", accuracy, correct, total);

        println!("\n=== METRICHE PER CLASSE ===");
        for (idx, name) in class_names.iter().enumerate() {
            let m = metrics.get(&idx).unwrap();
            println!("\n{} (classe {}):", name, idx);
            println!("  Precision: {:.4}", m.precision());
            println!("  Recall:    {:.4}", m.recall());
            println!("  F1-Score:  {:.4}", m.f1_score());
            println!(
                "  TP: {}, FP: {}, FN: {}, TN: {}",
                m.true_positives, m.false_positives, m.false_negatives, m.true_negatives
            );
        }

        // Calcola macro-average
        let avg_precision: f64 = metrics.values().map(|m| m.precision()).sum::<f64>() / 3.0;
        let avg_recall: f64 = metrics.values().map(|m| m.recall()).sum::<f64>() / 3.0;
        let avg_f1: f64 = metrics.values().map(|m| m.f1_score()).sum::<f64>() / 3.0;

        println!("\n=== METRICHE MEDIE (MACRO) ===");
        println!("Avg Precision: {:.4}", avg_precision);
        println!("Avg Recall:    {:.4}", avg_recall);
        println!("Avg F1-Score:  {:.4}", avg_f1);
        self
    }
}
#[derive(Default)]
pub struct ModelBuilder {
    features: Option<Array3<f32>>,
    targets: Option<Array3<f32>>,
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
            losses: vec![],
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
    pub fn set_targets(mut self, targets: Array3<f32>) -> Self {
        Self {
            targets: Some(targets),
            ..self
        }
    }
    pub fn set_features(mut self, features: Array3<f32>) -> Self {
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
