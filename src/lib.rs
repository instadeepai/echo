use pyo3::prelude::*;

pub mod array_spec;
pub mod ingress;
pub mod metrics;
mod py_bindings;
pub mod ring_buf;
pub mod selector;
pub mod store;
pub mod transport;

use py_bindings::{PyHistSnapshot, PySampleInfo, PyServer, PyTcpTransport};

#[pymodule]
fn echo(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyServer>()?;
    m.add_class::<PyTcpTransport>()?;
    m.add_class::<PySampleInfo>()?;
    m.add_class::<PyHistSnapshot>()?;
    Ok(())
}
