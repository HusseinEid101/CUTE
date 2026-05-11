//! PyO3 bindings: `cute_tokenizer._accel`.
//!
//! Thin facade over `cute-core`. The Python wrapper modules in
//! `src/cute_tokenizer/` import from here and gracefully fall back to
//! the pure-Python implementation when `CUTE_USE_PYTHON_PRETOKENIZER=1`
//! or when the extension fails to load.

// PyO3 0.22 emits `?`-style propagation inside the wrap_pyfunction!
// macro that clippy mis-flags as `useless_conversion` on functions whose
// return type is `PyResult<...>`. Suppressed crate-wide rather than
// per-call to keep call sites readable.
#![allow(clippy::useless_conversion)]

use std::path::PathBuf;

use ahash::AHashMap;
use cute_core::{decode, frequency, identifier, pretok, pua, secrets, tokens, ACCEL_BUILD_TAG};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

// ---------------------------------------------------------------------------
// PreparedMapping — reusable handle over a PUA mapping.
// ---------------------------------------------------------------------------

/// Cached Rust-side representation of a `PUAMapping`. Building the
/// `AHashMap`s once and reusing across calls amortizes the dict→hashmap
/// conversion cost over many encode/decode invocations.
#[pyclass(name = "PreparedMapping", module = "cute_tokenizer._accel", frozen)]
struct PreparedMapping {
    word_to_pua: pretok::WordToPua,
    pua_to_word: pretok::PuaToWord,
}

#[pymethods]
impl PreparedMapping {
    #[new]
    fn new(word_to_pua: &Bound<'_, PyDict>, pua_to_word: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut wtp: pretok::WordToPua = AHashMap::with_capacity(word_to_pua.len());
        for (k, v) in word_to_pua.iter() {
            let key: String = k.extract()?;
            let val: String = v.extract()?;
            let mut chars = val.chars();
            let ch = chars.next().ok_or_else(|| {
                PyValueError::new_err("word_to_pua value must be a non-empty string")
            })?;
            if chars.next().is_some() {
                return Err(PyValueError::new_err(
                    "word_to_pua value must be a single character",
                ));
            }
            wtp.insert(key, ch);
        }

        let mut ptw: pretok::PuaToWord = AHashMap::with_capacity(pua_to_word.len());
        for (k, v) in pua_to_word.iter() {
            let key_str: String = k.extract()?;
            let mut chars = key_str.chars();
            let ch = chars.next().ok_or_else(|| {
                PyValueError::new_err("pua_to_word key must be a non-empty string")
            })?;
            if chars.next().is_some() {
                return Err(PyValueError::new_err(
                    "pua_to_word key must be a single character",
                ));
            }
            let val: String = v.extract()?;
            ptw.insert(ch, val);
        }

        Ok(Self {
            word_to_pua: wtp,
            pua_to_word: ptw,
        })
    }

    fn __len__(&self) -> usize {
        self.word_to_pua.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "<PreparedMapping size={} pua_chars={}>",
            self.word_to_pua.len(),
            self.pua_to_word.len()
        )
    }
}

// ---------------------------------------------------------------------------
// Token-level helpers.
// ---------------------------------------------------------------------------

#[pyfunction]
fn iter_token_strings<'py>(py: Python<'py>, text: String) -> Bound<'py, PyList> {
    let collected: Vec<String> = py.allow_threads(|| {
        tokens::iter_token_strings(&text)
            .map(|s| s.to_string())
            .collect()
    });
    PyList::new_bound(py, &collected)
}

#[pyfunction]
fn is_identifier(s: &str) -> bool {
    identifier::is_identifier(s)
}

#[pyfunction]
fn split_identifier<'py>(py: Python<'py>, ident: String) -> Bound<'py, PyList> {
    let parts = py.allow_threads(|| identifier::split_identifier(&ident));
    PyList::new_bound(py, &parts)
}

#[pyfunction]
fn is_pua_char(s: &str) -> bool {
    let mut chars = s.chars();
    match (chars.next(), chars.next()) {
        (Some(ch), None) => pua::is_pua_char(ch),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// cute_split_text / pretokenize_to_string / reverse_pua_substitute.
// ---------------------------------------------------------------------------

#[pyfunction]
fn cute_split_text<'py>(
    py: Python<'py>,
    text: String,
    prepared: &PreparedMapping,
) -> Bound<'py, PyList> {
    let pieces = py.allow_threads(|| pretok::cute_split_text(&text, &prepared.word_to_pua));
    PyList::new_bound(py, &pieces)
}

#[pyfunction]
fn pretokenize_to_string(py: Python<'_>, text: String, prepared: &PreparedMapping) -> String {
    py.allow_threads(|| pretok::pretokenize_to_string(&text, &prepared.word_to_pua))
}

/// Batch pre-tokenization. Processes `texts` in parallel via Rayon,
/// returning a `list[str]` in the same order. Eliminates per-item Python
/// dispatch overhead that costs ~5µs per text in `_batch_encode_plus`.
#[pyfunction]
fn pretokenize_batch<'py>(
    py: Python<'py>,
    texts: Vec<String>,
    prepared: &PreparedMapping,
) -> Bound<'py, PyList> {
    let results: Vec<String> = py.allow_threads(|| {
        use rayon::prelude::*;
        texts
            .par_iter()
            .map(|t| pretok::pretokenize_to_string(t, &prepared.word_to_pua))
            .collect()
    });
    PyList::new_bound(py, &results)
}

#[pyfunction]
fn reverse_pua_substitute(py: Python<'_>, text: String, prepared: &PreparedMapping) -> String {
    py.allow_threads(|| decode::reverse_pua_substitute(&text, &prepared.pua_to_word))
}

/// Batch reverse PUA substitution. Same Rayon-parallel rationale as
/// `pretokenize_batch`.
#[pyfunction]
fn reverse_pua_batch<'py>(
    py: Python<'py>,
    texts: Vec<String>,
    prepared: &PreparedMapping,
) -> Bound<'py, PyList> {
    let results: Vec<String> = py.allow_threads(|| {
        use rayon::prelude::*;
        texts
            .par_iter()
            .map(|t| decode::reverse_pua_substitute(t, &prepared.pua_to_word))
            .collect()
    });
    PyList::new_bound(py, &results)
}

// ---------------------------------------------------------------------------
// Frequency counting.
// ---------------------------------------------------------------------------

/// Single-text counter — returns `(counter_delta, boost_delta)` as Python
/// dicts. Caller (Python `count_in_text` wrapper) merges into Counter /
/// defaultdict.
#[pyfunction]
fn count_in_text<'py>(
    py: Python<'py>,
    text: String,
    boost_weight: f64,
    max_token_len: usize,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let (counter, boost) = py.allow_threads(|| {
        let mut c: AHashMap<String, u64> = AHashMap::new();
        let mut b: AHashMap<String, f64> = AHashMap::new();
        frequency::count_in_text(&text, &mut c, &mut b, boost_weight, max_token_len);
        (c, b)
    });
    let py_c = PyDict::new_bound(py);
    for (k, v) in counter {
        py_c.set_item(k, v)?;
    }
    let py_b = PyDict::new_bound(py);
    for (k, v) in boost {
        py_b.set_item(k, v)?;
    }
    Ok((py_c, py_b))
}

/// Parallel shard frequency counter — replaces the Python
/// `ProcessPoolExecutor` path. Reads gzipped JSONL shards in Rust and
/// aggregates counts via Rayon work-stealing.
#[pyfunction]
fn count_frequencies<'py>(
    py: Python<'py>,
    shard_paths: Vec<PathBuf>,
    boost_weight: f64,
    max_token_len: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let merged = py
        .allow_threads(|| frequency::count_frequencies(&shard_paths, boost_weight, max_token_len))
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
    let py_d = PyDict::new_bound(py);
    for (k, v) in merged {
        py_d.set_item(k, v)?;
    }
    Ok(py_d)
}

// ---------------------------------------------------------------------------
// Secret pattern scanner.
// ---------------------------------------------------------------------------

#[pyfunction]
fn has_secret(py: Python<'_>, text: String) -> Option<&'static str> {
    py.allow_threads(|| secrets::has_secret(&text))
}

// ---------------------------------------------------------------------------
// cute-bpe — purpose-built BPE encoder (1.1.0 hot path).
// ---------------------------------------------------------------------------

#[pyclass(name = "BPEEncoder", module = "cute_tokenizer._accel", frozen)]
struct PyBPEEncoder {
    inner: cute_bpe::CuteBpe,
}

#[pymethods]
impl PyBPEEncoder {
    #[new]
    fn new(tokenizer_json: &str, cute_mapping_json: &str) -> PyResult<Self> {
        let inner = cute_bpe::CuteBpe::from_paths(
            std::path::Path::new(tokenizer_json),
            std::path::Path::new(cute_mapping_json),
        )
        .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn encode<'py>(&self, py: Python<'py>, text: String) -> Bound<'py, PyList> {
        let ids: Vec<u32> = py.allow_threads(|| self.inner.encode(&text));
        PyList::new_bound(py, &ids)
    }

    fn encode_batch<'py>(&self, py: Python<'py>, texts: Vec<String>) -> Bound<'py, PyList> {
        use rayon::prelude::*;
        let all: Vec<Vec<u32>> =
            py.allow_threads(|| texts.par_iter().map(|t| self.inner.encode(t)).collect());
        let out = PyList::empty_bound(py);
        for ids in all {
            let inner = PyList::new_bound(py, &ids);
            out.append(inner).expect("append");
        }
        out
    }

    fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> String {
        py.allow_threads(|| self.inner.decode(&ids))
    }

    fn decode_batch<'py>(&self, py: Python<'py>, ids_list: Vec<Vec<u32>>) -> Bound<'py, PyList> {
        use rayon::prelude::*;
        let all: Vec<String> = py.allow_threads(|| {
            ids_list
                .par_iter()
                .map(|ids| self.inner.decode(ids))
                .collect()
        });
        PyList::new_bound(py, &all)
    }

    fn vocab_size(&self) -> u32 {
        self.inner.vocab_size()
    }
}

// ---------------------------------------------------------------------------
// Module init.
// ---------------------------------------------------------------------------

#[pymodule]
fn _accel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__build_tag__", ACCEL_BUILD_TAG)?;
    m.add_class::<PreparedMapping>()?;
    m.add_class::<PyBPEEncoder>()?;
    m.add_function(wrap_pyfunction!(iter_token_strings, m)?)?;
    m.add_function(wrap_pyfunction!(is_identifier, m)?)?;
    m.add_function(wrap_pyfunction!(split_identifier, m)?)?;
    m.add_function(wrap_pyfunction!(is_pua_char, m)?)?;
    m.add_function(wrap_pyfunction!(cute_split_text, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenize_to_string, m)?)?;
    m.add_function(wrap_pyfunction!(pretokenize_batch, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_pua_substitute, m)?)?;
    m.add_function(wrap_pyfunction!(reverse_pua_batch, m)?)?;
    m.add_function(wrap_pyfunction!(count_in_text, m)?)?;
    m.add_function(wrap_pyfunction!(count_frequencies, m)?)?;
    m.add_function(wrap_pyfunction!(has_secret, m)?)?;
    Ok(())
}
