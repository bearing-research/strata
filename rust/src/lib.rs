//! Rust acceleration for Strata's data plane.
//!
//! Narrow scope: `read_file_bytes`, an mmap-based cache read called from the cache-hit fast
//! path in `cache.py` for files at or above `fast_io.MMAP_MIN_BYTES`.
//!
//! Everything else lives in Python / PyArrow: those libraries are already C++ under the hood,
//! and merging IPC streams has to stay incremental (bounded memory), which PyArrow does.

use memmap2::Mmap;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::fs::File;

/// Read a file's bytes through mmap (for cache passthrough).
///
/// Args:
///     path: Path to the file
///
/// Returns:
///     bytes: Raw file contents
#[pyfunction]
fn read_file_bytes<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyBytes>> {
    let file = File::open(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| PyIOError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &mmap[..]))
}

/// Python module definition
#[pymodule]
#[pyo3(name = "_strata_core")]
fn strata_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_file_bytes, m)?)?;
    Ok(())
}
