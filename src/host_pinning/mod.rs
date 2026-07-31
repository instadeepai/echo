//! Optional CUDA host-memory pinning (page-locking) of the ring buffers.
//!
//! Copying out of pageable memory is not a DMA transfer: the driver stages it
//! through a small internal pinned buffer in CPU-executed chunks, holding the
//! calling thread and the driver lock throughout. Page-locking makes the same
//! copy one descriptor on the copy engine.
//!
//! Either every buffer ends up locked or the caller gets an error saying why;
//! there is no best-effort path, because a silent no-op is indistinguishable
//! from pinning not helping. See `docs/src/guides/host-memory-pinning.md`.
//!
//! The runtime is `dlopen`ed at pin time, so there is no build-time or
//! link-time CUDA dependency and nothing loads unless pinning is requested.
//!
//! Two jobs, one submodule each: [`resolve`] finds the runtime, [`register`]
//! page-locks memory with it. This root holds only what both need — the entry
//! points, the regions they act on, and the error either can return.

mod register;
mod resolve;

pub(crate) use register::{pin_all, unpin_all};
pub(crate) use resolve::api;

use std::fmt;
use std::os::raw::{c_char, c_int, c_uint, c_void};

/// `cudaError_t cudaHostRegister(void*, size_t, unsigned int)`
type RegisterFn = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> c_int;
/// `cudaError_t cudaHostUnregister(void*)`
type UnregisterFn = unsafe extern "C" fn(*mut c_void) -> c_int;
/// `const char* cudaGetErrorName(cudaError_t)`
type ErrorNameFn = unsafe extern "C" fn(c_int) -> *const c_char;
/// `cudaError_t cudaFree(void*)`
type FreeFn = unsafe extern "C" fn(*mut c_void) -> c_int;

const CUDA_SUCCESS: c_int = 0;

/// The CUDA entry points pinning needs.
///
/// Passed explicitly rather than reached through a global so tests can inject
/// stubs and assert rollback without a GPU.
#[derive(Clone, Copy)]
pub(crate) struct CudaApi {
    register: RegisterFn,
    unregister: UnregisterFn,
    error_name: ErrorNameFn,
    free: FreeFn,
}

/// A contiguous host allocation to page-lock. Owns nothing; the validity
/// invariants live on [`pin_all`] / [`unpin_all`].
#[derive(Clone, Copy)]
pub(crate) struct Region {
    pub ptr: *mut u8,
    pub len: usize,
}

/// Why pinning could not be delivered. On failure nothing is left registered.
#[derive(Debug)]
pub(crate) enum PinError {
    /// No CUDA runtime could be loaded. One line per probed path, so a caller
    /// can fix their environment without reading this source.
    RuntimeUnavailable { probed: Vec<String> },
    /// The runtime rejected a registration. `name` is the CUDA error symbol.
    Registration {
        name: String,
        code: c_int,
        /// Which of the caller's regions was rejected.
        region_index: usize,
        len: usize,
    },
}

impl fmt::Display for PinError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PinError::RuntimeUnavailable { probed } => {
                write!(
                    f,
                    "pin_host_memory=True but no CUDA runtime could be loaded. Probed, in order:"
                )?;
                for line in probed {
                    write!(f, "\n  - {line}")?;
                }
                write!(
                    f,
                    "\nInstall a CUDA runtime (for example the nvidia-cuda-runtime wheel), or \
                     construct the Server after your framework has initialised CUDA."
                )
            }
            PinError::Registration {
                name,
                code,
                region_index,
                len,
            } => write!(
                f,
                "pin_host_memory=True but cudaHostRegister failed with {name} ({code}) on ring \
                 buffer {region_index} ({len} bytes). Registrations made before the failure have been \
                 rolled back. A usable CUDA device must be visible to this process; construct the \
                 Server after CUDA is initialised."
            ),
        }
    }
}

impl std::error::Error for PinError {}
