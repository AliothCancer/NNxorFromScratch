pub mod node;
pub mod neural_network;

use std::collections::HashMap;
use std::error::Error;

use ndarray::{  s, Array1};
use polars::prelude::*;

fn run() -> Result<(), Box<dyn Error>> {
    let file_path = "iris_dataset.csv"; // Inserisci il percorso del tuo file CSV
    let (features, target) = load_iris_dataset(file_path)?;
    // Visualizza i primi 5 record per verifica
    println!("Features:\n{}", features.head(Some(5)));
    println!("Target:\n{}", target.slice(s![0..5]));

    Ok(())
}

fn _get_different_values(df: &DataFrame, col_name: &str) {
    let col = df.column(col_name).expect("accessing col");

    let mut values = vec![];
    col.iter().for_each(|value| {
        //dbg!(value);
        if !values.contains(&value) {
            values.push(value);
        }
    });
    println!(
        "{:?}",
        values
            .iter()
            .map(|x| x.get_str().unwrap())
            .collect::<Vec<_>>()
    );
}



fn load_iris_dataset(file_path: &str) -> Result<(DataFrame, Array1<f64>), Box<dyn Error>> {
    // Leggi il file CSV in un DataFrame di Polars
    let df = CsvReadOptions::default()
        .try_into_reader_with_file_path(Some(file_path.into()))
        .unwrap()
        .finish()?;

    // Mappatura dei nomi delle specie a numeri
    let mut species_map = HashMap::new();
    species_map.insert("Iris-setosa", 0.0);
    species_map.insert("Iris-versicolor", 1.0);
    species_map.insert("Iris-virginica", 2.0);

    // Converti la colonna 'target' in numeri utilizzando la mappatura
    let target_series = df
        .column("target")?
        .str()?
        .into_iter()
        .map(|opt_str| {
            opt_str
                .and_then(|s| species_map.get(s).copied())
                .ok_or_else(|| "Unknown species".to_string())
        })
        .collect::<Result<Vec<f64>, String>>()?;

    // Converti i vettori in Array1
    let target_array = Array1::from_vec(target_series);

    // Seleziona le colonne di input (caratteristiche)
    let features = df.select(&[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ])?;

    Ok((features, target_array))
}
