//! Tests for registration, rollback and teardown.
//!
//! The stubbed half runs anywhere — rollback's only visible consequence is the
//! *absence* of leaked registrations, which cannot be observed from Python, so it
//! is checked through the injected API. The rest needs a real GPU and reports a
//! skip when there isn't one.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;
use std::path::PathBuf;

use super::*;
use crate::host_pinning::resolve::{api, scan_mapped_runtimes};
use crate::host_pinning::RegisterFn;

// --- registration and rollback, with the CUDA calls stubbed out ---

thread_local! {
    static REGISTERED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static UNREGISTERED: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static INITIALISED: RefCell<usize> = const { RefCell::new(0) };
}

/// Fails on the third region, succeeds on every other.
unsafe extern "C" fn stub_register(ptr: *mut c_void, _len: usize, _flags: c_uint) -> c_int {
    let nth = REGISTERED.with(|c| {
        c.borrow_mut().push(ptr as usize);
        c.borrow().len()
    });
    if nth == 3 {
        1 // cudaErrorInvalidValue
    } else {
        CUDA_SUCCESS
    }
}

unsafe extern "C" fn stub_register_ok(ptr: *mut c_void, _len: usize, _flags: c_uint) -> c_int {
    REGISTERED.with(|c| c.borrow_mut().push(ptr as usize));
    CUDA_SUCCESS
}

unsafe extern "C" fn stub_unregister(ptr: *mut c_void) -> c_int {
    UNREGISTERED.with(|c| c.borrow_mut().push(ptr as usize));
    CUDA_SUCCESS
}

unsafe extern "C" fn stub_error_name(_code: c_int) -> *const c_char {
    c"cudaErrorInvalidValue".as_ptr()
}

unsafe extern "C" fn stub_free(_ptr: *mut c_void) -> c_int {
    INITIALISED.with(|c| *c.borrow_mut() += 1);
    CUDA_SUCCESS
}

fn stub_api(register: RegisterFn) -> CudaApi {
    REGISTERED.with(|c| c.borrow_mut().clear());
    UNREGISTERED.with(|c| c.borrow_mut().clear());
    INITIALISED.with(|c| *c.borrow_mut() = 0);
    CudaApi {
        register,
        unregister: stub_unregister,
        error_name: stub_error_name,
        free: stub_free,
    }
}

/// Dangling, but never dereferenced: only passed to the stub API.
fn fake_regions(n: usize) -> Vec<Region> {
    (1..=n)
        .map(|i| Region {
            ptr: (i * 0x1000) as *mut u8,
            len: 4096,
        })
        .collect()
}

#[test]
fn a_failure_on_the_third_region_rolls_back_exactly_the_first_two() {
    let api = stub_api(stub_register);
    let regions = fake_regions(5);

    // SAFETY: the stub API never dereferences these pointers.
    let err = unsafe { pin_all(&api, &regions) }.expect_err("registration should have failed");

    let unregistered = UNREGISTERED.with(|c| c.borrow().clone());
    assert_eq!(
        unregistered,
        vec![0x1000, 0x2000],
        "rollback must unregister the regions that succeeded, and only those"
    );
    // Nothing past the failure is attempted.
    assert_eq!(REGISTERED.with(|c| c.borrow().len()), 3);
    assert!(
        err.to_string().contains("cudaErrorInvalidValue"),
        "the CUDA error must be named, not numeric: {err}"
    );
}

#[test]
fn a_successful_pin_registers_every_region_and_rolls_back_nothing() {
    let api = stub_api(stub_register_ok);
    let regions = fake_regions(4);

    // SAFETY: the stub API never dereferences these pointers.
    unsafe { pin_all(&api, &regions) }.expect("registration should have succeeded");

    assert_eq!(
        REGISTERED.with(|c| c.borrow().clone()),
        vec![0x1000, 0x2000, 0x3000, 0x4000]
    );
    assert!(UNREGISTERED.with(|c| c.borrow().is_empty()));
}

#[test]
fn pinning_forces_runtime_initialisation_first() {
    let api = stub_api(stub_register_ok);
    // SAFETY: the stub API never dereferences these pointers.
    unsafe { pin_all(&api, &fake_regions(1)) }.unwrap();
    assert_eq!(INITIALISED.with(|c| *c.borrow()), 1);
}

#[test]
fn dropping_a_pinned_ring_buffer_unregisters_every_buffer() {
    // Teardown without a GPU: real ring buffer, stubbed CUDA. A leak here
    // would be invisible from Python, hence the injection point.
    let api = stub_api(stub_register_ok);
    let mut ring = crate::ring_buf::PytreeRingBuf::new(vec![64, 128], 8, 4);
    ring.pin_with(api)
        .expect("stubbed registration should succeed");

    let registered = REGISTERED.with(|c| c.borrow().clone());
    assert_eq!(registered.len(), 2, "one registration per array");
    assert!(UNREGISTERED.with(|c| c.borrow().is_empty()));

    drop(ring);
    assert_eq!(
        UNREGISTERED.with(|c| c.borrow().clone()),
        registered,
        "drop must unregister exactly what was registered"
    );
}

#[test]
fn dropping_an_unpinned_ring_buffer_touches_no_cuda_entry_point() {
    let _api = stub_api(stub_register_ok);
    drop(crate::ring_buf::PytreeRingBuf::new(vec![64], 8, 4));
    assert!(REGISTERED.with(|c| c.borrow().is_empty()));
    assert!(UNREGISTERED.with(|c| c.borrow().is_empty()));
}

#[test]
fn a_ring_buffer_whose_registration_fails_leaves_nothing_registered() {
    // The third array is rejected, so the two that took must be unregistered
    // and `Drop` must then do nothing more.
    let api = stub_api(stub_register);
    let mut ring = crate::ring_buf::PytreeRingBuf::new(vec![64, 64, 64, 64], 8, 4);
    ring.pin_with(api).expect_err("the third array should fail");

    let after_rollback = UNREGISTERED.with(|c| c.borrow().clone());
    assert_eq!(after_rollback.len(), 2);
    drop(ring);
    assert_eq!(
        UNREGISTERED.with(|c| c.borrow().clone()),
        after_rollback,
        "a failed pin must leave Drop with nothing to reverse"
    );
}

#[test]
fn unpin_all_unregisters_every_region() {
    let api = stub_api(stub_register_ok);
    // SAFETY: the stub API never dereferences these pointers.
    unsafe { unpin_all(&api, &fake_regions(3)) };
    assert_eq!(
        UNREGISTERED.with(|c| c.borrow().clone()),
        vec![0x1000, 0x2000, 0x3000]
    );
}

// --- against a real CUDA runtime, when the machine has one ---

/// The dev checkout's CUDA vendor package: `cargo test` has no Python
/// interpreter to ask. See `docs/src/development.md` for installing the wheel.
fn dev_vendor_roots() -> Vec<PathBuf> {
    let venv = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".venv/lib");
    let Ok(entries) = std::fs::read_dir(venv) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|e| e.path().join("site-packages/nvidia"))
        .filter(|p| p.is_dir())
        .collect()
}

/// `cudaError_t cudaHostGetFlags(unsigned int*, void*)`
type HostGetFlagsFn = unsafe extern "C" fn(*mut c_uint, *mut c_void) -> c_int;

/// The runtime, or `None` on a machine without one. Rust has no native test
/// skip, so these report and pass rather than fail on CPU-only CI.
fn real_api() -> Option<&'static CudaApi> {
    match api(&dev_vendor_roots()) {
        Ok(api) => Some(api),
        Err(e) => {
            eprintln!("skipping: no CUDA runtime on this machine\n{e}");
            None
        }
    }
}

/// `cudaHostGetFlags`, looked up separately from the [`CudaApi`] under test:
/// going through those pointers would only prove they believe they succeeded.
///
/// Call after resolution has run, so this re-opens the mapped runtime
/// (`RTLD_NOLOAD`) rather than loading a second copy.
fn host_get_flags() -> HostGetFlagsFn {
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();
    let mapped = scan_mapped_runtimes(&maps)
        .into_iter()
        .next()
        .expect("resolution has run, so a runtime is mapped");
    let path = CString::new(mapped).unwrap();
    unsafe {
        let handle = libc::dlopen(
            path.as_ptr(),
            libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NOLOAD,
        );
        assert!(!handle.is_null(), "the mapped runtime should re-open");
        let sym = libc::dlsym(handle, c"cudaHostGetFlags".as_ptr());
        assert!(!sym.is_null(), "the runtime should export cudaHostGetFlags");
        std::mem::transmute::<*mut c_void, HostGetFlagsFn>(sym)
    }
}

#[test]
fn a_real_runtime_confirms_the_buffers_are_registered_portable() {
    let Some(api) = real_api() else { return };
    let mut buffer = vec![0u8; 4 << 20];
    let region = Region {
        ptr: buffer.as_mut_ptr(),
        len: buffer.len(),
    };

    // SAFETY: `buffer` outlives the registration and is unregistered below.
    unsafe { pin_all(api, std::slice::from_ref(&region)) }
        .expect("registration failed on a GPU host");

    let mut flags: c_uint = 0;
    let code = unsafe { (host_get_flags())(&mut flags, region.ptr as *mut c_void) };
    assert_eq!(code, CUDA_SUCCESS, "cudaHostGetFlags rejected the address");
    assert_eq!(
        flags & CUDA_HOST_REGISTER_PORTABLE,
        CUDA_HOST_REGISTER_PORTABLE
    );

    // SAFETY: just registered above, through this same api.
    unsafe { unpin_all(api, std::slice::from_ref(&region)) };
}

#[test]
fn a_real_ring_buffer_pins_and_unregisters_on_drop() {
    if real_api().is_none() {
        return;
    }
    let roots = dev_vendor_roots();
    let mut ring = crate::ring_buf::PytreeRingBuf::new(vec![1024, 2048], 64, 8);
    ring.pin_host_memory(&roots)
        .expect("registration failed on a GPU host");

    let (address, _) = ring.range_ptr(0, 0, 8);
    let mut flags: c_uint = 0;
    let code = unsafe { (host_get_flags())(&mut flags, address as *mut c_void) };
    assert_eq!(code, CUDA_SUCCESS);
    assert_eq!(
        flags & CUDA_HOST_REGISTER_PORTABLE,
        CUDA_HOST_REGISTER_PORTABLE
    );

    // Drop unregisters, so the runtime must no longer know the address.
    drop(ring);
    let code = unsafe { (host_get_flags())(&mut flags, address as *mut c_void) };
    assert_ne!(
        code, CUDA_SUCCESS,
        "drop should have unregistered the ring buffer"
    );
}

#[test]
fn pinning_twice_in_one_process_reuses_the_resolved_runtime() {
    // Two servers in one process, no extra configuration.
    if real_api().is_none() {
        return;
    }
    let roots = dev_vendor_roots();
    let mut first = crate::ring_buf::PytreeRingBuf::new(vec![4096], 32, 4);
    let mut second = crate::ring_buf::PytreeRingBuf::new(vec![4096], 32, 4);
    first.pin_host_memory(&roots).expect("first ring failed");
    second.pin_host_memory(&roots).expect("second ring failed");
}
