use linear_regression::LinearRegression;

pub mod linear_regression;

fn main() {
    let x_data = (0..10_000).map(|x| x as f64).collect::<Vec<_>>();
    let y_data = x_data.iter().map(|x| x * 2.0 + 3.0).collect::<Vec<_>>();

    let x_mean = x_data.iter().sum::<f64>() / x_data.len() as f64;
    let x_std =
        (x_data.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>() / x_data.len() as f64).sqrt();
    let x_data: Vec<f64> = x_data.iter().map(|x| (x - x_mean) / x_std).collect();

    let y_mean = y_data.iter().sum::<f64>() / y_data.len() as f64;
    let y_std =
        (y_data.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>() / y_data.len() as f64).sqrt();
    let y_data: Vec<f64> = y_data.iter().map(|y| (y - y_mean) / y_std).collect();


    
    let mut model = LinearRegression::new();
    model
        // Initializing
        .set_bias(0.0)
        .set_learning_rate(0.0001)
        .set_slope(0.0)
        .set_iteration(100_000)
        .set_x(x_data.clone())
        .set_y(y_data)
        // Loop
        .run_training();

    let x = 25.5;
    let y_pred = model.predict(&x);

    println!("x = {}  ->  y = {}", x, y_pred);
}
