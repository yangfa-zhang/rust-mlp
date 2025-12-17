use pyo3::prelude::*;
use pyo3::types::PyModule;

#[pyclass]
pub struct Number(i32);

#[pymethods]
impl Number {
    #[new]
    fn new(value: i32) -> Self {
        Number(value)
    }
}

