#![allow(clippy::let_and_return)]

use ndarray::{arr2, Array2, ArrayView2};

#[inline]
pub fn loss(y_hat: &Array2<f32>, y_true: ArrayView2<f32>) -> Array2<f32> {
    // dbg!(&y_true);
    match y_true[(0, 0)] {
        0.0 => {
            let k =-(arr2(&[[1.0]]) - y_hat).ln();
            // dbg!(&y_hat,&k);
            k
        },
        1.0 => {
            let k =-y_hat.ln();
            // dbg!(&y_hat,&k);
            k
        },
        _ => unreachable!(),
    }
}

#[inline]
/// La derivata combinata di BCE e Sigmoid si semplifica in: y_hat - y_true
/// Questo è un trucco matematico molto utile!
pub fn bce_and_sigmoid_grad(y_hat: &Array2<f32>, y_true: ArrayView2<f32>) -> Array2<f32> {
    y_hat.to_owned() - y_true
}

#[inline]
pub fn sigmoid(v: &Array2<f32>) -> Array2<f32> {
    v.mapv(|v|1.0 / (1.0 + (-v).exp()))
}


#[inline]
pub fn tanh(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|x|x.tanh())
}

#[inline]
pub fn tanh_grad(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|x|1.0 - x.tanh().powi(2))
}

