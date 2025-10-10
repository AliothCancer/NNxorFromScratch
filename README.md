It offers a simple api to train a user defined custom neural network with currently some limitation:

```rust
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
}
```

Training on XOR input and output.

```
Inizio training XOR...

Epoch 0: Loss mean= 0.9237
Epoch 100: Loss mean= 0.6215
Epoch 200: Loss mean= 0.3994
Epoch 300: Loss mean= 0.1442
Epoch 400: Loss mean= 0.0722
Epoch 500: Loss mean= 0.0463
Epoch 600: Loss mean= 0.0338
Epoch 700: Loss mean= 0.0265
Epoch 800: Loss mean= 0.0218
Epoch 900: Loss mean= 0.0185

=== FINAL RESULTS ===
Input: [0, 0] -> Predizione: 0.0139, Atteso: 0
Input: [1, 1] -> Predizione: 0.0151, Atteso: 0
Input: [0, 1] -> Predizione: 0.9698, Atteso: 1
Input: [1, 0] -> Predizione: 0.9958, Atteso: 1
```
