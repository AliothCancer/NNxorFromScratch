use crate::neural_net::layer::{Layer, LayerType};

type NodesQuantity = usize;

/// This struct will build all layers at once, based on architecture
/// it will return a Vec<Layer> 
#[derive(Default)]
pub struct LayersBuilder {
    layers_architecture: Vec<(LayerType, NodesQuantity)>,
}

impl LayersBuilder {
    pub fn new(layers_architecture: Vec<(LayerType, NodesQuantity)>)-> LayersBuilder{
        Self { layers_architecture }
    }

    pub fn build(self) -> Vec<Layer> {
        let mut layers = vec![];
        for (layer_type, nodes_quantity) in self.layers_architecture{
            todo!()
        }
        layers
    }
}
