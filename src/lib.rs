#![allow(non_snake_case)]

use pyo3::prelude::*;
use pyo3::exceptions::PyRuntimeError;
use pyo3::types::PyModule;
use pyo3::wrap_pymodule;


use candle_core::{DType, Device, Tensor};
use candle_nn::{
    linear, Linear, Module, VarBuilder, VarMap, Optimizer,
};

pub mod base;   
pub mod mlp;


#[pymodule]
fn rust_mlp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(mlp::mlp))?;
    Ok(())
}