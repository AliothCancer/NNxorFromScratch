#[derive(Debug, Clone)]
pub struct LinearRegression {
    x: Vec<f64>,
    y: Vec<f64>,
    slope: f64,
    bias: f64,
    learning_rate: f64,
    iteration: usize,
}

impl Default for LinearRegression {
    fn default() -> Self {
        Self {
            slope: Default::default(),
            bias: Default::default(),
            learning_rate: 0.01,
            x: Vec::new(),
            y: Vec::new(),
            iteration: 100,
        }
    }
}

// INITIALIZING
impl LinearRegression {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_slope(&mut self, slope: f64) -> &mut LinearRegression {
        self.slope = slope;
        self
    }

    pub fn set_bias(&mut self, bias: f64) -> &mut LinearRegression {
        self.bias = bias;
        self
    }

    pub fn set_learning_rate(&mut self, learning_rate: f64) -> &mut LinearRegression {
        self.learning_rate = learning_rate;
        self
    }

    pub fn set_iteration(&mut self, iteration: usize) -> &mut LinearRegression {
        self.iteration = iteration;
        self
    }

    pub fn set_x(&mut self, x: Vec<f64>) -> &mut Self {
        self.x = x;
        self
    }

    pub fn set_y(&mut self, y: Vec<f64>) -> &mut Self {
        self.y = y;
        self
    }
}

// Calculation methods
impl LinearRegression {
    /// Return the prediction of y based on x with current model parameters
    pub fn predict(&self, x: &f64) -> f64 {
        // y = mx + q
        self.slope * x + self.bias
    }

    fn calculate_mse(&self) -> f64 {
        self.x
            .iter()
            .zip(self.y.iter())
            .map(|(x, y)| {
                //dbg!(self.predict(x));
                (y - self.predict(x)).powf(2.0)
            })
            .sum::<f64>()
            / self.x.len() as f64
    }

    fn calculate_mse_gradient(&self) -> (f64, f64) {
        let n = self.x.len() as f64;

        let (dm, db) = self
            .x
            .iter()
            .zip(self.y.iter())
            .map(|(x, y)| {
                let error = y - (self.slope * x + self.bias);
                let dm = -2.0 * x * error;
                let db = -2.0 * error;
                (dm, db)
            })
            .fold((0.0, 0.0), |(acc_dm, acc_db), (dm, db)| {
                (acc_dm + dm, acc_db + db)
            });

        (dm / n, db / n)
    }
    fn update_params(&mut self, gradient: (f64, f64)) {
        let (dm, db) = gradient;
        self.slope -= self.learning_rate * dm;
        self.bias -= self.learning_rate * db;
        //dbg!(self.slope);
        //dbg!(self.bias);
    }
}

// TRAINING LOOP
impl LinearRegression {
    pub fn run_training(&mut self) -> &LinearRegression {
        println!("Starting training");
        (0..self.iteration).for_each(|_| self.train_cicle());
        self
    }

    fn train_cicle(&mut self) {
        //let mse = self.calculate_mse();
        //dbg!(mse);
        let gradient = self.calculate_mse_gradient();
        //dbg!(gradient); // Aggiungi questo
        self.update_params(gradient);
        //dbg!(self.slope, self.bias); // E questo
    }
}
