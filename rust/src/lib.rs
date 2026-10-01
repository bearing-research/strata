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

    let total_size: usize = segments.iter().map(|s| s.len()).sum();
    let mut result = Vec::with_capacity(total_size);

    // Keep the first segment's schema but strip its EOS marker (last 8 bytes).
    let first = &segments[0];
    if first.len() < 8 {
        return Err(StrataError::InvalidFile("First segment too small".into()));
    }

    if &first[first.len() - 8..] != &EOS_MARKER {
        return Err(StrataError::InvalidFile(
            "First segment missing EOS marker".into(),
        ));
    }

    result.extend_from_slice(&first[..first.len() - 8]);

    // Later segments: skip the schema message and copy only record batches.
    //
    // Malformed shapes return Err, never skip: Ok with nothing copied silently drops
    // rows, while Err hands the input to the Arrow fallback, which parses it or fails.
    for segment in &segments[1..] {
        // Empty segments are legitimate (fast_io's pyarrow path skips them too); a
        // nonempty one too small for an EOS marker is corrupt.
        if segment.is_empty() {
            continue;
        }
        if segment.len() < 8 {
            return Err(StrataError::InvalidFile(
                "segment too small to contain an EOS marker".into(),
            ));
        }

        if &segment[segment.len() - 8..] != &EOS_MARKER {
            return Err(StrataError::InvalidFile(
                "Segment missing EOS marker".into(),
            ));
        }

        // Schema message: continuation (4) + size (4) + flatbuffer (size bytes)
        let mut offset = 0;

        // Skip continuation marker if present
        if segment.len() >= 4 && &segment[0..4] == &CONTINUATION_MARKER {
            offset = 4;
        }

        if offset + 4 > segment.len() {
            return Err(StrataError::InvalidFile(
                "segment truncated before its schema-message length".into(),
            ));
        }
        let schema_size = u32::from_le_bytes([
            segment[offset],
            segment[offset + 1],
            segment[offset + 2],
            segment[offset + 3],
        ]) as usize;

        // Checked: schema_size comes off disk, and wrapping would turn a bogus length
        // into a plausible in-range offset.
        offset = offset
            .checked_add(4)
            .and_then(|o| o.checked_add(schema_size))
            .and_then(|o| o.checked_add(7))
            .ok_or_else(|| {
                StrataError::InvalidFile("schema-message length overflows the segment".into())
            })?;

        // Align to 8 bytes (the +7 above is folded into the checked chain)
        offset &= !7;

        // Record batches run from the end of the schema message to the EOS marker.
        let batches_end = segment.len() - 8;
        if offset > batches_end {
            return Err(StrataError::InvalidFile(
                "schema-message length runs past the segment's record batches".into(),
            ));
        }
        // offset == batches_end is a schema-only segment: nothing to copy.
        if offset < batches_end {
            result.extend_from_slice(&segment[offset..batches_end]);
        }
    }

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

        // Non-contiguous buffer: copy into owned memory so the concat path sees a plain
        // slice. ``to_vec(py)?`` raises if the object cannot export a u8 buffer at all.
        Ok(Self::Owned(buffer.to_vec(py)?))
    }

    fn as_slice<'py>(&'py self, py: Python<'py>) -> PyResult<&'py [u8]> {
        match self {
            Self::Backed(bytes) => Ok(bytes.as_ref()),
            Self::Buffer(buffer) => {
                let cells = buffer.as_slice(py).ok_or_else(|| {
                    PyValueError::new_err("bytes-like segment must be C-contiguous")
                })?;
                // SAFETY: ReadOnlyCell<u8> is repr(transparent) over a byte cell. We hold the
                // PyBuffer for the slice's lifetime and never call into Python while it is in
                // use, so the memory stays valid for this FFI call.
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

    match concat_streams_fast(&segment_slices) {
        Ok(result) => return Ok(PyBytes::new(py, &result)),
        Err(_) => {
            // Fall back to full Arrow parsing (slower but handles edge cases)
        }
    }

    if segment_slices.is_empty() {
        return Ok(PyBytes::new(py, &[]));
    }

    let first_cursor = Cursor::new(segment_slices[0]);
    let first_reader = StreamReader::try_new(first_cursor, None).map_err(StrataError::from)?;
    let schema = first_reader.schema();

    let mut all_batches = Vec::new();

    for segment in &segment_slices {
        let cursor = Cursor::new(segment);
        let reader = StreamReader::try_new(cursor, None).map_err(StrataError::from)?;

        for batch_result in reader {
            let batch = batch_result.map_err(StrataError::from)?;
            all_batches.push(batch);
        }
    }

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
