use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;

pub trait Model {
    fn train(&mut self, x: Vec<Vec<f32>>, y: Vec<f32>, epochs: usize) -> PyResult<()>;
    fn evaluate(&self, x: Vec<Vec<f32>>, y: Vec<f32>) -> PyResult<()>;
}