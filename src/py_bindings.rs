use std::sync::Arc;

use numpy::npyffi::{
    flags::{NPY_ARRAY_ALIGNED, NPY_ARRAY_C_CONTIGUOUS},
    types::{npy_intp, NPY_TYPES},
    NpyTypes, PY_ARRAY_API,
};
use numpy::PyArray1;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::array_spec::ArraySpec;
use crate::ingress::DrainerPool;
use crate::metrics::Metrics;
use crate::selector::{FifoRemover, FifoSampler};
use crate::store::Store;
use crate::transport::{self, TcpTransport};

#[pyclass(name = "TcpTransport", frozen)]
pub struct PyTcpTransport {
    port: u16,
    num_threads: usize,
}

#[pymethods]
impl PyTcpTransport {
    #[new]
    #[pyo3(signature = (port, num_threads = 8))]
    fn new(port: u16, num_threads: usize) -> Self {
        Self { port, num_threads }
    }
}

/// Histogram percentiles (ns) exposed to Python. Mirrors
/// `metrics::HistSnapshot`.
#[pyclass(name = "HistSnapshot", frozen, get_all, skip_from_py_object)]
#[derive(Clone, Copy, Default)]
pub struct PyHistSnapshot {
    pub count: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: f64,
    pub p50_ns: u64,
    pub p90_ns: u64,
    pub p99_ns: u64,
}

impl From<crate::metrics::HistSnapshot> for PyHistSnapshot {
    fn from(h: crate::metrics::HistSnapshot) -> Self {
        Self {
            count: h.count,
            min_ns: h.min_ns,
            max_ns: h.max_ns,
            mean_ns: h.mean_ns,
            p50_ns: h.p50_ns,
            p90_ns: h.p90_ns,
            p99_ns: h.p99_ns,
        }
    }
}

/// Snapshot of backpressure + throughput metrics returned with every sample.
///
/// The histogram fields and CAS counters are populated only when the
/// `detailed-metrics` build feature is enabled; otherwise they come back as
/// zero-valued `HistSnapshot`s / zero counters.
#[pyclass(name = "SampleInfo", frozen, get_all)]
pub struct PySampleInfo {
    // --- always on (health) ---
    pub store_size: usize,
    pub active_connections: usize,
    pub queue_depth_sum: usize,
    pub queue_depth_max: usize,
    pub push_blocked_count: u64,
    pub samples_inserted_total: u64,
    pub notify_count: u64,

    // --- detailed (zero unless feature enabled) ---
    pub cas_success_total: u64,
    pub cas_failure_total: u64,
    pub reservation_retries_total: u64,
    pub drain_round: PyHistSnapshot,
    pub memcpy: PyHistSnapshot,
    pub queue_dwell: PyHistSnapshot,
}

/// Return type of `_Server.sample()`: per-array uint8 views + a SampleInfo.
type SampleOut<'py> = (Vec<Bound<'py, PyArray1<u8>>>, Py<PySampleInfo>);

#[pyclass(name = "_Server")]
pub struct PyServer {
    store: Arc<Store>,
    drainer_pool: Option<Arc<DrainerPool>>,
    transport: Option<Box<dyn transport::Transport>>,
    metrics: Arc<Metrics>,
}

#[pymethods]
impl PyServer {
    #[new]
    #[pyo3(signature = (shapes, dtype_sizes, batch_size, transport=None, num_buffers=3, num_drainers=8, producer_queue_size=8))]
    fn new(
        shapes: Vec<Vec<usize>>,
        dtype_sizes: Vec<usize>,
        batch_size: usize,
        transport: Option<&Bound<'_, PyAny>>,
        num_buffers: usize,
        num_drainers: usize,
        producer_queue_size: usize,
    ) -> PyResult<Self> {
        if shapes.len() != dtype_sizes.len() {
            return Err(PyValueError::new_err(
                "shapes and dtype_sizes must have same length",
            ));
        }

        let specs: Vec<ArraySpec> = shapes
            .into_iter()
            .zip(dtype_sizes)
            .map(|(shape, dtype_size)| ArraySpec::new(shape, dtype_size))
            .collect();
        let transport_specs = specs.clone();

        let capacity = batch_size * num_buffers;
        let metrics = Metrics::new(num_drainers);
        let store = Arc::new(Store::new(
            specs,
            batch_size,
            num_buffers,
            Box::new(FifoSampler::with_metrics(
                batch_size,
                capacity,
                Some(metrics.clone()),
            )),
            Box::new(FifoRemover::new()),
        ));

        // Drainer pool + transport are only needed when there's a network
        // transport; in-process submit() skips both and writes directly.
        let (drainer_pool, boxed_transport) = match transport {
            Some(t) => {
                let pool = Arc::new(DrainerPool::new(
                    store.clone(),
                    num_drainers,
                    producer_queue_size,
                    metrics.clone(),
                ));
                let boxed: Box<dyn transport::Transport> =
                    if let Ok(tcp) = t.cast::<PyTcpTransport>() {
                        let tcp = tcp.borrow();
                        Box::new(TcpTransport::new(
                            tcp.num_threads,
                            pool.clone(),
                            transport_specs,
                            tcp.port,
                        ))
                    } else {
                        return Err(PyValueError::new_err("transport must be TcpTransport"));
                    };
                (Some(pool), Some(boxed))
            }
            None => (None, None),
        };

        Ok(Self {
            store,
            drainer_pool,
            transport: boxed_transport,
            metrics,
        })
    }

    /// Start the transport (bind port, begin accepting connections).
    fn start(&self) -> PyResult<()> {
        let t = self.transport.as_ref().ok_or_else(|| {
            PyRuntimeError::new_err("no transport configured; pass TcpTransport to __init__")
        })?;
        t.start()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to start transport: {e}")))
    }

    /// Block until a batch is ready.
    ///
    /// Returns `(arrays, info)` where `arrays` is a list of uint8 numpy views
    /// into Rust-owned memory (invalidated on the next `sample` call), and
    /// `info` is a `SampleInfo` snapshot of backpressure metrics. Returns None
    /// on shutdown.
    fn sample<'py>(&self, py: Python<'py>) -> PyResult<Option<SampleOut<'py>>> {
        let store = self.store.clone();
        let view = py.detach(move || store.sample());

        let cv = match view {
            Some(cv) => cv,
            None => return Ok(None),
        };

        let arrays: Vec<Bound<'py, PyArray1<u8>>> = cv
            .arrays
            .iter()
            .map(|&(addr, len)| unsafe { array_view_u8(py, addr, len) })
            .collect::<PyResult<_>>()?;

        let info = PySampleInfo {
            store_size: self.store.size(),
            active_connections: self.metrics.active_connections(),
            queue_depth_sum: self.metrics.queue_depth_sum(),
            queue_depth_max: self.metrics.queue_depth_max(),
            push_blocked_count: self.metrics.push_blocked_count(),
            samples_inserted_total: self.metrics.samples_inserted_total(),
            notify_count: self.metrics.notify_count(),
            cas_success_total: self.metrics.cas_success_total(),
            cas_failure_total: self.metrics.cas_failure_total(),
            reservation_retries_total: self.metrics.reservation_retries_total(),
            drain_round: self.metrics.drain_round_aggregate().into(),
            memcpy: self.metrics.memcpy_aggregate().into(),
            queue_dwell: self.metrics.queue_dwell_aggregate().into(),
        };
        let info = Py::new(py, info)?;

        Ok(Some((arrays, info)))
    }

    /// Submit raw bytes for a sample (for in-process use / testing / benchmarking).
    /// Releases the GIL so multiple Python threads can submit concurrently.
    fn submit(&self, py: Python<'_>, data: Vec<Vec<u8>>) {
        let store = self.store.clone();
        py.detach(move || {
            let slices: Vec<&[u8]> = data.iter().map(|v| v.as_slice()).collect();
            store.insert_sync(&slices);
        });
    }

    /// Reset the per-drainer timing histograms (memcpy, drain_round,
    /// queue_dwell). Counters and gauges are unchanged. Call at the boundary
    /// of a metrics window so the next `SampleInfo` reflects only events
    /// recorded after this call.
    fn reset_histograms(&self) {
        self.metrics.reset_histograms();
    }

    /// Signal shutdown: stops transport, drainer pool, then unblocks sample.
    fn shutdown(&self) {
        if let Some(t) = self.transport.as_ref() {
            t.shutdown();
        }
        if let Some(pool) = self.drainer_pool.as_ref() {
            pool.shutdown();
        }
        self.store.shutdown();
    }
}

/// Create a 1-D uint8 numpy array that is a view into existing memory.
/// The array does NOT own the data (NPY_ARRAY_OWNDATA is not set).
/// The caller must ensure the memory remains valid for the array's lifetime.
unsafe fn array_view_u8<'py>(
    py: Python<'py>,
    addr: usize,
    len: usize,
) -> PyResult<Bound<'py, PyArray1<u8>>> {
    let subtype = PY_ARRAY_API.get_type_object(py, NpyTypes::PyArray_Type);
    let mut dims = [len as npy_intp];
    let flags = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
    let arr_ptr = PY_ARRAY_API.PyArray_New(
        py,
        subtype,
        1,
        dims.as_mut_ptr(),
        NPY_TYPES::NPY_UBYTE as _,
        std::ptr::null_mut(), // strides -- null = C-contiguous
        addr as *mut std::ffi::c_void,
        1, // itemsize (bytes per element for uint8)
        flags,
        std::ptr::null_mut(), // obj
    );
    if arr_ptr.is_null() {
        return Err(PyRuntimeError::new_err("failed to create numpy view"));
    }
    Ok(Bound::from_owned_ptr(py, arr_ptr).cast_into_unchecked())
}
