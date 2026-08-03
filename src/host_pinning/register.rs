//! Page-locking host memory, and rolling back cleanly when part of it fails.

use std::ffi::CStr;
use std::os::raw::{c_int, c_uint, c_void};

use super::{CudaApi, PinError, Region, CUDA_SUCCESS};

/// `cudaHostRegisterPortable`: valid in every CUDA context in the process,
/// including ones created later. This is why pinning takes no device argument
/// and why N servers across N GPUs in one process need no configuration.
pub const CUDA_HOST_REGISTER_PORTABLE: c_uint = 0x01;

/// Page-lock every region, or leave none of them locked.
///
/// On failure the regions registered so far are unregistered first, so nothing
/// accumulates across retries. Normal teardown is `Drop`'s job.
///
/// # Safety
/// - Every region's `ptr` must be valid for `len` bytes.
/// - That allocation must not be reallocated, resized, or moved while it stays
///   registered — the registration pins the physical pages behind *these*
///   addresses.
/// - Each region registered here must be passed to [`unpin_all`] before its
///   memory is freed.
pub unsafe fn pin_all(api: &CudaApi, regions: &[Region]) -> Result<(), PinError> {
    // Registration against an uninitialised runtime fails, so force init.
    // Freeing null frees nothing and never changes the current device, but it is
    // not free: it creates the primary context on the current device, costing
    // that context's memory (~128 MB measured). Constructing after the framework
    // has initialised CUDA — the documented order — means it already exists.
    //
    // The result is ignored: portable registration makes the device irrelevant,
    // so the only failure worth reporting is registration's own, below.
    unsafe { (api.free)(std::ptr::null_mut()) };

    for (index, region) in regions.iter().enumerate() {
        if region.len == 0 {
            continue;
        }
        // SAFETY: the caller guarantees `ptr` is valid for `len` bytes.
        let code = unsafe {
            (api.register)(
                region.ptr as *mut c_void,
                region.len,
                CUDA_HOST_REGISTER_PORTABLE,
            )
        };
        if code != CUDA_SUCCESS {
            // SAFETY: these regions were just registered by this loop, so they
            // satisfy `unpin_all`'s contract.
            unsafe { unpin_all(api, &regions[..index]) };
            return Err(PinError::Registration {
                name: error_name(api, code),
                code,
                region_index: index,
                len: region.len,
            });
        }
    }
    Ok(())
}

/// Reverse of [`pin_all`]; must run before the memory is freed.
///
/// # Safety
/// Every region must currently be registered, by a [`pin_all`] call through
/// this same `api`, and must still be valid for `len` bytes.
pub unsafe fn unpin_all(api: &CudaApi, regions: &[Region]) {
    for region in regions {
        if region.len == 0 {
            continue;
        }
        // Errors are dropped: this runs from `Drop` and during rollback, and
        // there is nothing useful either could do about a failure.
        // SAFETY: the caller guarantees the region is registered and valid.
        unsafe { (api.unregister)(region.ptr as *mut c_void) };
    }
}

/// A CUDA error as its symbol (`cudaErrorInvalidValue`), so it can be looked up
/// rather than decoded from an integer.
fn error_name(api: &CudaApi, code: c_int) -> String {
    let name = unsafe { (api.error_name)(code) };
    if name.is_null() {
        return format!("unrecognised CUDA error {code}");
    }
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}
