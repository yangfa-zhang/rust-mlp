#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;

use candle_core::{DType, Device, Tensor};
use candle_nn::{
    linear, Linear, Module, VarBuilder, VarMap, Optimizer,
};
pub mod base;   
pub mod mlp;
use mlp::MLP;
#[pymodule]
fn rust_mlp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MLP>()?;
    Ok(())
}