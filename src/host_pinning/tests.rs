//! Unit tests for the private `host_pinning` module.
//!
//! In-crate rather than under `tests/`: the module is private, the stub-injection
//! tests need the internal `CudaApi`, and the resolution ladder is not public
//! surface.

use super::*;
use std::cell::RefCell;

// --- rung 1: parsing the process's own mapped files ---

/// A runtime mapped as several segments, unrelated libraries, anonymous and
/// special mappings, a path with a space in it, and a deleted mapping.
const MAPS_FIXTURE: &str = "\
55a3c0000000-55a3c0021000 r--p 00000000 fd:01 1179651                    /usr/bin/python3.11
7f1a00000000-7f1a00021000 r--p 00000000 fd:01 2359310                    /usr/lib/x86_64-linux-gnu/libc.so.6
7f1a10000000-7f1a10800000 rw-p 00000000 00:00 0
7f1a20000000-7f1a20a00000 r--p 00000000 fd:01 4194313                    /venv/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so.13
7f1a20a00000-7f1a21400000 r-xp 00a00000 fd:01 4194313                    /venv/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so.13
7f1a30000000-7f1a30100000 r-xp 00000000 fd:01 4194320                    /opt/my libs/libcudart.so.12
7f1a40000000-7f1a40100000 r-xp 00000000 fd:01 4194321                    /tmp/stale/libcudart.so.11 (deleted)
7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0                          [stack]
ffffffffff600000-ffffffffff601000 --xp 00000000 00:00 0                  [vsyscall]
";

#[test]
fn scan_finds_each_mapped_runtime_once_in_order() {
    assert_eq!(
        scan_mapped_runtimes(MAPS_FIXTURE),
        vec![
            "/venv/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so.13",
            "/opt/my libs/libcudart.so.12",
            "/tmp/stale/libcudart.so.11",
        ]
    );
}

#[test]
fn scan_ignores_unrelated_libraries() {
    let maps = "\
7f1a00000000-7f1a00021000 r-xp 00000000 fd:01 1 /usr/lib/libcudnn.so.9
7f1a10000000-7f1a10021000 r-xp 00000000 fd:01 2 /usr/lib/libcublas.so.13
7f1a20000000-7f1a20021000 r-xp 00000000 fd:01 3 /usr/lib/libcudart_static.a
";
    assert!(scan_mapped_runtimes(maps).is_empty());
}

#[test]
fn scan_of_a_process_with_no_runtime_yields_nothing() {
    assert!(scan_mapped_runtimes("").is_empty());
}

// --- rung 2: searching beneath the CUDA vendor package ---

fn touch(path: &std::path::Path) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, b"").unwrap();
}

#[test]
fn wheel_search_finds_the_consolidated_layout() {
    // CUDA 13: one wheel, `nvidia/cu13/lib/`.
    let root = tempfile::tempdir().unwrap();
    let nvidia = root.path().join("nvidia");
    touch(&nvidia.join("cu13/lib/libcudart.so.13"));
    touch(&nvidia.join("cu13/lib/libcudart_static.a"));
    touch(&nvidia.join("cudnn/lib/libcudnn.so.9"));

    assert_eq!(
        search_wheel_roots(std::slice::from_ref(&nvidia)),
        vec![nvidia
            .join("cu13/lib/libcudart.so.13")
            .to_string_lossy()
            .into_owned()]
    );
}

#[test]
fn wheel_search_finds_the_per_component_layout() {
    // CUDA 12: one wheel per component, `nvidia/cuda_runtime/lib/`.
    let root = tempfile::tempdir().unwrap();
    let nvidia = root.path().join("nvidia");
    touch(&nvidia.join("cuda_runtime/lib/libcudart.so.12"));
    touch(&nvidia.join("cublas/lib/libcublas.so.12"));

    assert_eq!(
        search_wheel_roots(std::slice::from_ref(&nvidia)),
        vec![nvidia
            .join("cuda_runtime/lib/libcudart.so.12")
            .to_string_lossy()
            .into_owned()]
    );
}

#[test]
fn wheel_search_prefers_the_newest_major_version() {
    let root = tempfile::tempdir().unwrap();
    let nvidia = root.path().join("nvidia");
    touch(&nvidia.join("cuda_runtime/lib/libcudart.so.9"));
    touch(&nvidia.join("cu13/lib/libcudart.so.13"));
    touch(&nvidia.join("cuda_runtime/lib/libcudart.so.12"));

    let found = search_wheel_roots(&[nvidia]);
    let names: Vec<&str> = found
        .iter()
        .map(|p| p.rsplit('/').next().unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["libcudart.so.13", "libcudart.so.12", "libcudart.so.9"]
    );
}

#[test]
fn wheel_search_tolerates_missing_and_empty_roots() {
    let root = tempfile::tempdir().unwrap();
    assert!(search_wheel_roots(&[]).is_empty());
    assert!(search_wheel_roots(&[root.path().join("does-not-exist")]).is_empty());
    assert!(search_wheel_roots(&[root.path().to_path_buf()]).is_empty());
}

// --- ladder ordering ---

#[test]
fn ladder_tries_mapped_then_wheel_then_sonames() {
    let root = tempfile::tempdir().unwrap();
    let nvidia = root.path().join("nvidia");
    touch(&nvidia.join("cu13/lib/libcudart.so.13"));
    let wheel_lib = nvidia
        .join("cu13/lib/libcudart.so.13")
        .to_string_lossy()
        .into_owned();

    let ladder = candidates(MAPS_FIXTURE, &[nvidia]);
    let rungs: Vec<Rung> = ladder.iter().map(|c| c.rung).collect();
    let names: Vec<&str> = ladder.iter().map(|c| c.name.as_str()).collect();

    assert_eq!(
        rungs,
        vec![
            Rung::AlreadyLoaded,
            Rung::AlreadyLoaded,
            Rung::AlreadyLoaded,
            Rung::InstalledWheel,
            Rung::Soname,
            Rung::Soname,
            Rung::Soname,
        ]
    );
    assert_eq!(
        names,
        vec![
            "/venv/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so.13",
            "/opt/my libs/libcudart.so.12",
            "/tmp/stale/libcudart.so.11",
            wheel_lib.as_str(),
            // Versioned before unversioned: the wheels ship no unversioned
            // symlink, which is what broke this before.
            "libcudart.so.13",
            "libcudart.so.12",
            "libcudart.so",
        ]
    );
}

#[test]
fn ladder_probes_each_path_once() {
    // A mapped runtime that is also the one the wheel ships.
    let root = tempfile::tempdir().unwrap();
    let nvidia = root.path().join("nvidia");
    let lib = nvidia.join("cu13/lib/libcudart.so.13");
    touch(&lib);
    let maps = format!(
        "7f1a20000000-7f1a20a00000 r-xp 0 fd:01 1 {}\n",
        lib.display()
    );

    let ladder = candidates(&maps, &[nvidia]);
    let mut names: Vec<&str> = ladder.iter().map(|c| c.name.as_str()).collect();
    let before = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), before, "ladder contains a duplicate path");
}

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
        let sym = symbol(handle, c"cudaHostGetFlags")
            .expect("the runtime should export cudaHostGetFlags");
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

#[test]
fn unresolvable_runtime_reports_every_path_probed() {
    let err = PinError::RuntimeUnavailable {
        probed: vec![
            "/a/libcudart.so.13: boom".into(),
            "libcudart.so: nope".into(),
        ],
    };
    let message = err.to_string();
    assert!(message.contains("/a/libcudart.so.13: boom"), "{message}");
    assert!(message.contains("libcudart.so: nope"), "{message}");
}
