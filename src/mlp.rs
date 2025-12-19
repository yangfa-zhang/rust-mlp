use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    linear, Linear, Module, VarBuilder, VarMap, Optimizer,
};
use crate::base::Model;
#[pyclass]
pub struct PyMLP {
    model: SimpleNN,
    optimizer: candle_nn::AdamW,
    device: Device,
}

#[pymethods]
impl PyMLP {
    #[new]
    fn new(input_dim: usize, lr: f64) -> PyResult<Self> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = SimpleNN::new(input_dim, vb)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let optimizer = candle_nn::AdamW::new_lr(varmap.all_vars(), lr)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(Self { model, optimizer, device })
    }

    fn train(&mut self, x: Vec<Vec<f32>>, y: Vec<f32>, epochs: usize) -> PyResult<()> {
        Model::train(self, x, y, epochs)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }

    fn evaluate(&self, x: Vec<Vec<f32>>, y: Vec<f32>) -> PyResult<()> {
        Model::evaluate(self, x, y)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok(())
    }
}
impl Model for PyMLP {
    fn train(&mut self, x: Vec<Vec<f32>>, y: Vec<f32>, epochs: usize) -> PyResult<()> {
        let n = x.len();
        let d = x[0].len();
        let x_flat = x.into_iter().flatten().collect::<Vec<_>>();
        let x_tensor = Tensor::from_vec(x_flat, (n, d), &self.device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let y_tensor = Tensor::from_vec(y, (n, 1), &self.device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        for _ in 0..epochs {
            let output = self.model.forward(&x_tensor)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            let loss = candle_nn::loss::mse(&output, &y_tensor)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            println!("Loss: {}", loss.to_scalar::<f32>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?);
            self.optimizer.backward_step(&loss)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    fn evaluate(&self, x: Vec<Vec<f32>>, y: Vec<f32>) -> PyResult<()> {
        let n = x.len();
        let d = x[0].len();
        let x_flat = x.into_iter().flatten().collect::<Vec<_>>();
        let x_tensor = Tensor::from_vec(x_flat, (n, d), &self.device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let y_tensor = Tensor::from_vec(y, (n, 1), &self.device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let output = self.model.forward(&x_tensor)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let loss = candle_nn::loss::mse(&output, &y_tensor)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        println!("Evaluation Loss: {}", loss.to_scalar::<f32>()
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?);
        Ok(())
    }
}


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

#[pymodule]
pub fn mlp(m: &Bound<'_, PyModule>)-> PyResult<()>{
    m.add_class::<PyMLP>()?;
    Ok(())
}