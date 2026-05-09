//! Rust acceleration for Strata's data plane.
//!
//! Narrow scope: two functions that sit on genuine hot paths.
//!
//! 1. `read_file_bytes` — mmap-based cache read, called from the
//!    cache hit fast path in `cache.py`.
//! 2. `concat_ipc_streams` — byte-level concatenation of Arrow IPC
//!    streams, skipping Arrow deserialize/reserialize. Used for
//!    buffered multi-row-group responses.
//!
//! Everything else lives in Python / PyArrow — those libraries are
//! already C++ under the hood and Rust wouldn't add value.
//!
//! Arrow IPC Stream format: schema + [record batches] + EOS marker.

use arrow::ipc::reader::StreamReader;
use arrow::ipc::writer::StreamWriter;
use memmap2::Mmap;
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::types::{PyAny, PyBytes};
use std::fs::File;
use std::io::Cursor;
use thiserror::Error;

// Arrow IPC constants
const CONTINUATION_MARKER: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
const EOS_MARKER: [u8; 8] = [0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00];

#[derive(Error, Debug)]
pub enum StrataError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[error("Invalid file: {0}")]
    InvalidFile(String),
}

impl From<StrataError> for PyErr {
    fn from(err: StrataError) -> PyErr {
        match err {
            StrataError::Io(e) => PyIOError::new_err(e.to_string()),
            StrataError::Arrow(e) => PyValueError::new_err(e.to_string()),
            StrataError::InvalidFile(msg) => PyValueError::new_err(msg),
        }
    }
}

/// Read an Arrow IPC file and return raw bytes (for cache passthrough).
///
/// Even simpler: just read the file bytes. Python can decide whether
/// to parse or pass through.
///
/// Args:
///     path: Path to the Arrow IPC file
///
/// Returns:
///     bytes: Raw file contents
#[pyfunction]
fn read_file_bytes<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyBytes>> {
    let file = File::open(path).map_err(StrataError::from)?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(StrataError::from)?;
    Ok(PyBytes::new(py, &mmap[..]))
}

/// Fast concatenation of Arrow IPC streams by byte manipulation.
///
/// Arrow IPC Stream format:
/// - Schema message (continuation + size + flatbuffer)
/// - Record batch messages (continuation + size + flatbuffer + data)
/// - EOS marker (0xFFFFFFFF 0x00000000)
///
/// To concatenate streams: take schema from first, strip schema from rest,
/// combine all record batches, add single EOS.
fn concat_streams_fast(segments: &[&[u8]]) -> Result<Vec<u8>, StrataError> {
    if segments.is_empty() {
        return Ok(Vec::new());
    }

    if segments.len() == 1 {
        return Ok(segments[0].to_vec());
    }

    // Estimate total size
    let total_size: usize = segments.iter().map(|s| s.len()).sum();
    let mut result = Vec::with_capacity(total_size);

    // Copy first segment completely (includes schema and EOS)
    // But we need to strip the EOS marker (last 8 bytes)
    let first = &segments[0];
    if first.len() < 8 {
        return Err(StrataError::InvalidFile("First segment too small".into()));
    }

    // Verify EOS marker at end of first segment
    if &first[first.len() - 8..] != &EOS_MARKER {
        return Err(StrataError::InvalidFile(
            "First segment missing EOS marker".into(),
        ));
    }

    // Copy first segment without EOS
    result.extend_from_slice(&first[..first.len() - 8]);

    // For subsequent segments, we need to skip the schema message and copy only record batches
    for segment in &segments[1..] {
        if segment.len() < 8 {
            continue;
        }

        // Verify EOS marker
        if &segment[segment.len() - 8..] != &EOS_MARKER {
            return Err(StrataError::InvalidFile(
                "Segment missing EOS marker".into(),
            ));
        }

        // Find where record batches start by parsing message headers
        // Schema message format: continuation (4) + size (4) + flatbuffer (size bytes)
        let mut offset = 0;

        // Skip continuation marker if present
        if segment.len() >= 4 && &segment[0..4] == &CONTINUATION_MARKER {
            offset = 4;
        }

        // Read schema message size
        if offset + 4 > segment.len() {
            continue;
        }
        let schema_size = u32::from_le_bytes([
            segment[offset],
            segment[offset + 1],
            segment[offset + 2],
            segment[offset + 3],
        ]) as usize;
        offset += 4 + schema_size;

        // Align to 8 bytes
        offset = (offset + 7) & !7;

        // Copy record batches (everything from offset to len-8)
        if offset < segment.len() - 8 {
            result.extend_from_slice(&segment[offset..segment.len() - 8]);
        }
    }

    // Add final EOS marker
    result.extend_from_slice(&EOS_MARKER);

    Ok(result)
}

enum BytesLikeInput {
    Backed(PyBackedBytes),
    Buffer(PyBuffer<u8>),
    Owned(Vec<u8>),
}

impl BytesLikeInput {
    fn extract(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(bytes) = obj.extract::<PyBackedBytes>() {
            return Ok(Self::Backed(bytes));
        }

        let buffer = PyBuffer::<u8>::get(obj)?;
        if buffer.as_slice(py).is_some() {
            return Ok(Self::Buffer(buffer));
        }

        // Fallback for non-contiguous buffer-protocol inputs. This copies into
        // owned Rust memory so the rest of the concat path can operate on a
        // plain byte slice. If the object cannot be exported as a contiguous
        // u8 buffer at all, ``to_vec(py)?`` raises and the Python exception
        // bubbles back to the caller unchanged.
        Ok(Self::Owned(buffer.to_vec(py)?))
    }

    fn as_slice<'py>(&'py self, py: Python<'py>) -> PyResult<&'py [u8]> {
        match self {
            Self::Backed(bytes) => Ok(bytes.as_ref()),
            Self::Buffer(buffer) => {
                let cells = buffer.as_slice(py).ok_or_else(|| {
                    PyValueError::new_err("bytes-like segment must be C-contiguous")
                })?;
                // SAFETY: ReadOnlyCell<u8> is repr(transparent) over a single
                // byte cell. We hold the PyBuffer for the lifetime of the
                // returned slice and never call back into Python while the
                // slice is in use, so the backing memory stays valid for this
                // FFI call.
                Ok(unsafe { std::slice::from_raw_parts(cells.as_ptr().cast::<u8>(), cells.len()) })
            }
            Self::Owned(bytes) => Ok(bytes.as_slice()),
        }
    }
}

/// Concatenate multiple Arrow IPC stream segments into one.
///
/// When serving multiple row groups, we need to combine them into
/// a single IPC stream. This does it efficiently in Rust using
/// byte manipulation rather than full Arrow parsing.
///
/// Args:
///     segments: Iterable of bytes-like Arrow IPC stream segments
///         (``bytes``, ``bytearray``, or ``memoryview``)
///
/// Returns:
///     bytes: Single combined Arrow IPC stream
#[pyfunction]
fn concat_ipc_streams<'py>(
    py: Python<'py>,
    segments: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let mut extracted = Vec::new();
    for item in segments.try_iter()? {
        let item = item?;
        extracted.push(BytesLikeInput::extract(py, &item)?);
    }

    let segment_slices: Vec<&[u8]> = extracted
        .iter()
        .map(|segment| segment.as_slice(py))
        .collect::<PyResult<_>>()?;

    // Try fast path first (byte manipulation)
    match concat_streams_fast(&segment_slices) {
        Ok(result) => return Ok(PyBytes::new(py, &result)),
        Err(_) => {
            // Fall back to full Arrow parsing (slower but handles edge cases)
        }
    }

    if segment_slices.is_empty() {
        return Ok(PyBytes::new(py, &[]));
    }

    // Read first segment to get schema
    let first_cursor = Cursor::new(segment_slices[0]);
    let first_reader = StreamReader::try_new(first_cursor, None).map_err(StrataError::from)?;
    let schema = first_reader.schema();

    // Collect all batches
    let mut all_batches = Vec::new();

    for segment in &segment_slices {
        let cursor = Cursor::new(segment);
        let reader = StreamReader::try_new(cursor, None).map_err(StrataError::from)?;

        for batch_result in reader {
            let batch = batch_result.map_err(StrataError::from)?;
            all_batches.push(batch);
        }
    }

    // Write combined stream
    let estimated_size: usize = segment_slices.iter().map(|s| s.len()).sum();
    let mut buffer = Vec::with_capacity(estimated_size);

    {
        let mut writer = StreamWriter::try_new(&mut buffer, &schema).map_err(StrataError::from)?;

        for batch in all_batches {
            writer.write(&batch).map_err(StrataError::from)?;
        }
        writer.finish().map_err(StrataError::from)?;
    }

    Ok(PyBytes::new(py, &buffer))
}

/// Python module definition
#[pymodule]
#[pyo3(name = "_strata_core")]
fn strata_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_file_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(concat_ipc_streams, m)?)?;
    Ok(())
}
