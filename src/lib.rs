#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    linear, Linear, Module, VarBuilder, VarMap, Optimizer,
};

/// ------------------------------
/// Python module entry (PyO3 0.21+)
/// ------------------------------
#[pymodule]
fn rust_mlp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(let_me_try, m)?)?;
    Ok(())
}

/// ------------------------------
/// Python functions
/// ------------------------------
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn let_me_try() -> PyResult<()> {
    run_training()
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

/// ------------------------------
/// Inner Rust logic (anyhow world)
/// ------------------------------
fn run_training() -> anyhow::Result<()> {
    let device = Device::Cpu;

    let file = std::fs::File::open("fetch_california_housing.json")?;
    let reader = std::io::BufReader::new(file);
    let data: Data = serde_json::from_reader(reader)?;

    let train_d1 = data.X_train.len();
    let train_d2 = data.X_train[0].len();
    let test_d1 = data.X_test.len();
    let test_d2 = data.X_test[0].len();

    let x_train_vec = data.X_train.into_iter().flatten().collect::<Vec<_>>();
    let x_test_vec = data.X_test.into_iter().flatten().collect::<Vec<_>>();

    let y_train_vec = data.y_train;
    let y_test_vec = data.y_test;

    let x_train = Tensor::from_vec(x_train_vec, (train_d1, train_d2), &device)?;
    let y_train = Tensor::from_vec(y_train_vec, (train_d1, 1), &device)?;
    let x_test = Tensor::from_vec(x_test_vec, (test_d1, test_d2), &device)?;
    let y_test = Tensor::from_vec(y_test_vec, (test_d1, 1), &device)?;

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let model = SimpleNN::new(train_d2, vb)?;

    // ✅ candle-nn 0.9 正确写法
    let mut optimizer =
        candle_nn::AdamW::new_lr(varmap.all_vars(), 1e-2)?;

    train_model(&model, &x_train, &y_train, &mut optimizer, 200)?;
    evaluate_model(&model, &x_test, &y_test)?;

    Ok(())
}

/// ------------------------------
/// Model
/// ------------------------------
#[derive(Debug)]
struct SimpleNN {
    fc1: Linear,
    fc2: Linear,
}

impl SimpleNN {
    fn new(in_dim: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let fc1 = linear(in_dim, 64, vb.pp("fc1"))?;
        let fc2 = linear(64, 1, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }
}

impl Module for SimpleNN {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.fc1.forward(xs)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        Ok(x)
    }
}

/// ------------------------------
/// Training
/// ------------------------------
fn train_model(
    model: &SimpleNN,
    x_train: &Tensor,
    y_train: &Tensor,
    optimizer: &mut candle_nn::AdamW,
    epochs: usize,
) -> anyhow::Result<()> {
    for epoch in 0..epochs {
        let output = model.forward(x_train)?;
        let loss = candle_nn::loss::mse(&output, y_train)?;
        optimizer.backward_step(&loss)?;

        if (epoch + 1) % 10 == 0 {
            println!(
                "Epoch {} | Train Loss: {:.6}",
                epoch + 1,
                loss.to_scalar::<f32>()?
            );
        }
    }
    Ok(())
}

fn evaluate_model(
    model: &SimpleNN,
    x_test: &Tensor,
    y_test: &Tensor,
) -> anyhow::Result<()> {
    let output = model.forward(x_test)?;
    let loss = candle_nn::loss::mse(&output, y_test)?;
    println!("Test Loss: {:.6}", loss.to_scalar::<f32>()?);
    Ok(())
}

/// ------------------------------
/// Data
/// ------------------------------
#[derive(Debug, serde::Deserialize)]
struct Data {
    X_train: Vec<Vec<f32>>,
    X_test: Vec<Vec<f32>>,
    y_train: Vec<f32>,
    y_test: Vec<f32>,
}
