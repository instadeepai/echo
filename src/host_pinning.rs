//! Optional CUDA host-memory pinning (page-locking) of the ring buffers.
//!
//! Enabled at runtime by `ECHO_PIN_HOST_MEMORY=1` (off by default). When enabled,
//! [`pin`] registers a buffer with `cudaHostRegister` so a downstream
//! `jax.device_put` of its numpy view becomes a fast, truly-async H2D DMA instead
//! of a synchronous host->device staging copy. libcudart is resolved at runtime
//! via `dlopen`, so this carries NO CUDA toolchain / link-time dependency: the
//! code is always compiled but is a graceful no-op when the env var is unset, on
//! CUDA-less hosts, or if registration fails.

use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

// cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags)
type RegisterFn = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> c_int;
// cudaError_t cudaHostUnregister(void* ptr)
type UnregisterFn = unsafe extern "C" fn(*mut c_void) -> c_int;

struct Api {
    register: RegisterFn,
    unregister: UnregisterFn,
}
// The dlopen handle outlives the process; the resolved fn pointers are immutable.
unsafe impl Sync for Api {}
unsafe impl Send for Api {}

fn load() -> Option<Api> {
    // Opt-in: only attempt to pin when explicitly enabled.
    if std::env::var("ECHO_PIN_HOST_MEMORY").as_deref() != Ok("1") {
        return None;
    }
    unsafe {
        let handle = libc::dlopen(
            b"libcudart.so\0".as_ptr() as *const c_char,
            libc::RTLD_NOW | libc::RTLD_GLOBAL,
        );
        if handle.is_null() {
            return None;
        }
        let register = libc::dlsym(handle, b"cudaHostRegister\0".as_ptr() as *const c_char);
        let unregister = libc::dlsym(handle, b"cudaHostUnregister\0".as_ptr() as *const c_char);
        if register.is_null() || unregister.is_null() {
            return None;
        }
        Some(Api {
            register: std::mem::transmute::<*mut c_void, RegisterFn>(register),
            unregister: std::mem::transmute::<*mut c_void, UnregisterFn>(unregister),
        })
    }
}

fn api() -> Option<&'static Api> {
    static API: OnceLock<Option<Api>> = OnceLock::new();
    API.get_or_init(load).as_ref()
}

/// Page-lock `[ptr, ptr + len)` (portable across the process's CUDA contexts).
///
/// No-op unless `ECHO_PIN_HOST_MEMORY=1` and libcudart is present. Best-effort:
/// registration errors leave the memory pageable.
pub(crate) fn pin(ptr: *mut u8, len: usize) {
    if len == 0 {
        return;
    }
    if let Some(api) = api() {
        // cudaHostRegisterPortable = 0x01 (usable from every device/context).
        unsafe {
            (api.register)(ptr as *mut c_void, len, 0x01);
        }
    }
}

/// Reverse of [`pin`]; must run before the buffer is freed.
pub(crate) fn unpin(ptr: *mut u8, len: usize) {
    if len == 0 {
        return;
    }
    if let Some(api) = api() {
        unsafe {
            (api.unregister)(ptr as *mut c_void);
        }
    }
}
