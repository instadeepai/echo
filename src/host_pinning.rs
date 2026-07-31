//! Optional CUDA host-memory pinning (page-locking) of the ring buffers.
//!
//! A host-to-device copy out of *pageable* memory is not the DMA transfer it
//! looks like: the driver stages it through a small internal pinned buffer in
//! CPU-executed chunks — thousands of driver round-trips for a large batch,
//! which both moves bytes slower and holds the driver lock long enough to
//! starve the consumer thread's own kernel launches. Page-locking the ring
//! buffers turns the same copy into one DMA descriptor on the copy engine.
//!
//! Off unless the caller asks for it, and when asked this module either
//! delivers fully page-locked buffers or says why it could not. There is
//! deliberately no partial-success or best-effort path: a silent no-op is
//! indistinguishable from "pinning doesn't help this workload", and that
//! confusion is the defect this module was rewritten to remove.
//!
//! The runtime is resolved by `dlopen` at pin time, so echo carries no
//! build-time or link-time CUDA dependency and one wheel installs on GPU and
//! CPU-only hosts alike. Nothing here runs unless pinning is requested.

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int, c_uint, c_void};
use std::path::PathBuf;
use std::sync::OnceLock;

/// `cudaError_t cudaHostRegister(void*, size_t, unsigned int)`
type RegisterFn = unsafe extern "C" fn(*mut c_void, usize, c_uint) -> c_int;
/// `cudaError_t cudaHostUnregister(void*)`
type UnregisterFn = unsafe extern "C" fn(*mut c_void) -> c_int;
/// `const char* cudaGetErrorName(cudaError_t)`
type ErrorNameFn = unsafe extern "C" fn(c_int) -> *const c_char;
/// `cudaError_t cudaFree(void*)`
type FreeFn = unsafe extern "C" fn(*mut c_void) -> c_int;

const CUDA_SUCCESS: c_int = 0;

/// `cudaHostRegisterPortable`: the locked pages are valid in every CUDA context
/// in the process, including contexts created after registration.
///
/// This is why echo needs no device parameter and never selects a device. N
/// servers across N GPUs in one process each register portably, and every
/// registration is DMA-fast for every device regardless of which context
/// happened to be current at the time.
const CUDA_HOST_REGISTER_PORTABLE: c_uint = 0x01;

/// The CUDA entry points pinning needs, as plain function pointers.
///
/// Passed explicitly to [`pin_all`] / [`unpin_all`] rather than reached through
/// a process-global so that registration and rollback can be exercised with
/// stubs on a machine with no GPU — the rollback path is unsafe code whose only
/// externally visible consequence is the *absence* of leaked registrations.
#[derive(Clone, Copy)]
pub(crate) struct CudaApi {
    register: RegisterFn,
    unregister: UnregisterFn,
    error_name: ErrorNameFn,
    free: FreeFn,
}

/// A contiguous host allocation to page-lock.
///
/// Just a (pointer, length) pair — it owns nothing and asserts nothing. The
/// invariants that make registering it sound live on [`pin_all`] and
/// [`unpin_all`], which are `unsafe` for that reason.
#[derive(Clone, Copy)]
pub(crate) struct Region {
    pub ptr: *mut u8,
    pub len: usize,
}

/// Why pinning could not be delivered. Never returned for a partial success:
/// on failure nothing is left registered.
#[derive(Debug)]
pub(crate) enum PinError {
    /// No CUDA runtime could be loaded. Carries one line per probed path so a
    /// caller can fix their environment without reading echo's source.
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

// ---------------------------------------------------------------------------
// Resolving the CUDA runtime
// ---------------------------------------------------------------------------

/// Versioned sonames for the current and previous CUDA major versions, then the
/// unversioned one.
///
/// The unversioned soname is last and on its own is near-useless: the pip CUDA
/// runtime wheels ship only the versioned soname, with no unversioned symlink
/// and no ldconfig entry. Opening `libcudart.so` alone is what made this
/// feature a silent no-op for months.
const SONAMES: [&str; 3] = ["libcudart.so.13", "libcudart.so.12", "libcudart.so"];

/// Which rung of the resolution ladder produced a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rung {
    /// Rung 1: a runtime the process already has mapped.
    AlreadyLoaded,
    /// Rung 2: a runtime shipped by an installed CUDA wheel.
    InstalledWheel,
    /// Rung 3: a soname, left to the system loader.
    Soname,
}

impl Rung {
    fn label(self) -> &'static str {
        match self {
            Rung::AlreadyLoaded => "already-loaded scan",
            Rung::InstalledWheel => "installed-wheel search",
            Rung::Soname => "soname load",
        }
    }
}

/// One thing to hand to `dlopen`, in ladder order.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    /// An absolute path (rungs 1 and 2) or a bare soname (rung 3).
    name: String,
    rung: Rung,
}

/// True for a CUDA runtime shared object, by file name.
fn is_runtime_lib(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    // `libcudart.so`, `libcudart.so.13`, `libcudart.so.13.0.48` — but not
    // `libcudart_static.a`.
    name == "libcudart.so" || name.starts_with("libcudart.so.")
}

/// Rung 1: CUDA runtimes the process already has mapped, in first-seen order.
///
/// Version- and path-agnostic, and it hits whenever the framework has already
/// initialised CUDA, which is the common case. `dlopen`-ing the absolute path
/// of an already-mapped library returns a handle to that same mapping rather
/// than loading a second copy.
fn scan_mapped_runtimes(maps: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for line in maps.lines() {
        // The pathname is the last field of a `/proc/<pid>/maps` line and is
        // absolute, so it starts at the first " /" — found this way rather than
        // by splitting on whitespace because paths may contain spaces. An
        // unlinked file gets a " (deleted)" suffix.
        let line = line.strip_suffix(" (deleted)").unwrap_or(line);
        let Some(offset) = line.find(" /") else {
            continue;
        };
        let path = &line[offset + 1..];
        if is_runtime_lib(path) && !found.iter().any(|seen| seen == path) {
            found.push(path.to_string());
        }
    }
    found
}

/// The deepest a runtime library sits below the vendor package
/// (`nvidia/cu13/lib/libcudart.so.13` is three), plus room for a
/// reorganisation.
const WHEEL_SEARCH_DEPTH: usize = 4;

/// Rung 2: CUDA runtimes shipped by installed wheels, newest major first.
///
/// `roots` are the CUDA *vendor* package directories, located through Python's
/// import machinery by the caller. Searching the whole vendor package instead
/// of a named component subpackage is load-bearing, not lazy: CUDA 13 ships one
/// consolidated wheel laid out as `nvidia/cu13/lib/`, whereas CUDA 12 ships one
/// wheel per component laid out as `nvidia/cuda_runtime/lib/`. A lookup that
/// names the component finds nothing on a CUDA 13 install. Searching beneath
/// the vendor package covers both layouts and survives the next one.
///
/// This rung is what lets echo be constructed before the framework has loaded
/// CUDA, and it removes any need for loader-path environment variables.
fn search_wheel_roots(roots: &[PathBuf]) -> Vec<String> {
    let mut found: Vec<PathBuf> = Vec::new();
    let mut frontier: Vec<(PathBuf, usize)> = roots.iter().map(|r| (r.clone(), 0)).collect();

    while let Some((dir, depth)) = frontier.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue; // Missing or unreadable roots simply contribute nothing.
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            if file_type.is_dir() {
                if depth + 1 < WHEEL_SEARCH_DEPTH {
                    frontier.push((path, depth + 1));
                }
            } else if path.to_str().is_some_and(is_runtime_lib) {
                found.push(path);
            }
        }
    }

    // Newest major version first, then by path so the result is deterministic
    // regardless of directory iteration order.
    found.sort_by(|a, b| {
        let key = |p: &PathBuf| {
            let major = p
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(soname_major)
                .unwrap_or(0);
            std::cmp::Reverse(major)
        };
        key(a).cmp(&key(b)).then_with(|| a.cmp(b))
    });
    found
        .iter()
        .filter_map(|p| p.to_str().map(str::to_owned))
        .collect()
}

/// CUDA major version from a soname, e.g. `libcudart.so.13.0.48` -> 13.
fn soname_major(name: &str) -> Option<u32> {
    name.strip_prefix("libcudart.so.")?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// The full resolution ladder, in the order it will be tried.
///
/// Kept as one pure function of its inputs so that the ordering — the thing a
/// future edit could silently break, reintroducing the original bug — is
/// unit-testable without a GPU or a CUDA install.
fn candidates(maps: &str, vendor_roots: &[PathBuf]) -> Vec<Candidate> {
    let rungs = [
        (Rung::AlreadyLoaded, scan_mapped_runtimes(maps)),
        (Rung::InstalledWheel, search_wheel_roots(vendor_roots)),
        (
            Rung::Soname,
            SONAMES.iter().map(|s| (*s).to_owned()).collect(),
        ),
    ];

    let mut ladder: Vec<Candidate> = Vec::new();
    for (rung, names) in rungs {
        for name in names {
            // The same file can turn up on two rungs (a mapped runtime that is
            // also the one the wheel ships); probe it once, on the earlier rung.
            if !ladder.iter().any(|c| c.name == name) {
                ladder.push(Candidate { name, rung });
            }
        }
    }
    ladder
}

/// `dlopen` a candidate and resolve the four symbols pinning needs.
///
/// The handle is deliberately never `dlclose`d: the registrations it backs must
/// outlive it, and the runtime is process-wide state anyway.
fn open(candidate: &Candidate) -> Result<CudaApi, String> {
    let name = CString::new(candidate.name.as_str())
        .map_err(|_| "path contains an interior NUL byte".to_string())?;

    // RTLD_LOCAL rather than RTLD_GLOBAL: echo should not change how any other
    // library in the process resolves its symbols.
    let flags = libc::RTLD_NOW | libc::RTLD_LOCAL;
    let handle = unsafe {
        // Rung 3 asks the loader for an already-present library first, so a
        // bare soname prefers a copy the process has (under whatever path)
        // over pulling in a second one from the loader path.
        let noload = if candidate.rung == Rung::Soname {
            libc::dlopen(name.as_ptr(), flags | libc::RTLD_NOLOAD)
        } else {
            std::ptr::null_mut()
        };
        if noload.is_null() {
            libc::dlopen(name.as_ptr(), flags)
        } else {
            noload
        }
    };
    if handle.is_null() {
        return Err(dlerror().unwrap_or_else(|| "dlopen failed".to_string()));
    }

    // SAFETY: each symbol is transmuted to the signature libcudart declares for
    // it; a name that resolves in libcudart has that signature by definition.
    unsafe {
        Ok(CudaApi {
            register: std::mem::transmute::<*mut c_void, RegisterFn>(symbol(
                handle,
                c"cudaHostRegister",
            )?),
            unregister: std::mem::transmute::<*mut c_void, UnregisterFn>(symbol(
                handle,
                c"cudaHostUnregister",
            )?),
            error_name: std::mem::transmute::<*mut c_void, ErrorNameFn>(symbol(
                handle,
                c"cudaGetErrorName",
            )?),
            free: std::mem::transmute::<*mut c_void, FreeFn>(symbol(handle, c"cudaFree")?),
        })
    }
}

/// # Safety
/// `handle` must be a live handle returned by `dlopen`.
unsafe fn symbol(handle: *mut c_void, name: &CStr) -> Result<*mut c_void, String> {
    let sym = libc::dlsym(handle, name.as_ptr());
    if sym.is_null() {
        return Err(format!("opened, but {} is missing", name.to_string_lossy()));
    }
    Ok(sym)
}

fn dlerror() -> Option<String> {
    let err = unsafe { libc::dlerror() };
    if err.is_null() {
        return None;
    }
    Some(
        unsafe { CStr::from_ptr(err) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// Walk the ladder, returning the first runtime that opens or an error naming
/// every path probed.
fn resolve(vendor_roots: &[PathBuf]) -> Result<CudaApi, PinError> {
    // Absent (macOS, a container without procfs) just means rung 1 finds
    // nothing.
    let maps = std::fs::read_to_string("/proc/self/maps").unwrap_or_default();

    let ladder = candidates(&maps, vendor_roots);
    let mut probed = Vec::new();
    for rung in [Rung::AlreadyLoaded, Rung::InstalledWheel, Rung::Soname] {
        let of_rung: Vec<&Candidate> = ladder.iter().filter(|c| c.rung == rung).collect();
        // A rung that produced no candidate at all would otherwise be invisible,
        // leaving a reader unable to tell "searched and found nothing" from
        // "never ran".
        if of_rung.is_empty() {
            probed.push(empty_rung_note(rung, vendor_roots));
            continue;
        }
        for candidate in of_rung {
            match open(candidate) {
                Ok(api) => return Ok(api),
                Err(reason) => {
                    probed.push(format!("{} [{}]: {reason}", candidate.name, rung.label()))
                }
            }
        }
    }
    Err(PinError::RuntimeUnavailable { probed })
}

/// What to report for a rung that contributed no candidate, so the error says
/// where it looked rather than staying silent.
fn empty_rung_note(rung: Rung, vendor_roots: &[PathBuf]) -> String {
    let detail = match rung {
        Rung::AlreadyLoaded => "this process has no CUDA runtime mapped".to_string(),
        Rung::InstalledWheel if vendor_roots.is_empty() => {
            "no CUDA vendor package is importable".to_string()
        }
        Rung::InstalledWheel => format!(
            "no runtime found beneath {}",
            vendor_roots
                .iter()
                .map(|root| root.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Rung::Soname => "no soname candidates".to_string(),
    };
    format!("(none) [{}]: {detail}", rung.label())
}

static API: OnceLock<CudaApi> = OnceLock::new();

/// The CUDA runtime, resolving it on first use.
///
/// Only ever called when a caller asked for pinning, so the default-off path
/// loads nothing. A failure is not cached: a process that constructs a server
/// before its framework has initialised CUDA and retries later should get the
/// later, better answer.
pub(crate) fn api(vendor_roots: &[PathBuf]) -> Result<&'static CudaApi, PinError> {
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let resolved = resolve(vendor_roots)?;
    // Two threads racing here both resolve; `dlopen` is idempotent and
    // reference-counted, so the loser just drops an identical set of pointers.
    Ok(API.get_or_init(|| resolved))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Page-lock every region, or leave none of them locked.
///
/// On failure the regions registered so far are unregistered before returning,
/// so a caller whose construction fails leaves nothing behind for a retry or a
/// long-lived process to accumulate. Normal teardown is `Drop`'s job.
///
/// # Safety
/// - Every region's `ptr` must be valid for `len` bytes.
/// - That allocation must not be reallocated, resized, or moved while it stays
///   registered — the registration pins the physical pages behind *these*
///   addresses.
/// - Each region registered here must be passed to [`unpin_all`] before its
///   memory is freed.
pub(crate) unsafe fn pin_all(api: &CudaApi, regions: &[Region]) -> Result<(), PinError> {
    // Force runtime initialisation, best-effort: registration against an
    // uninitialised runtime fails, and this is what gives the runtime a reason
    // to set itself up. Freeing a null pointer frees nothing and does not
    // *change* the current device — but note it is not free of consequence: it
    // creates the primary CUDA context on whatever device is already current,
    // which costs that context's device memory (order of 100 MB). When the
    // caller follows the documented order and constructs the server after the
    // framework has initialised CUDA, the context already exists and this costs
    // nothing.
    //
    // The result is ignored on purpose: portable registration makes the choice
    // of device irrelevant, so the only failure that matters is registration's
    // own, which reports the real CUDA error a few lines below.
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
pub(crate) unsafe fn unpin_all(api: &CudaApi, regions: &[Region]) {
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

/// A CUDA error as its symbol (`cudaErrorInvalidValue`), so a reader can look
/// it up instead of decoding an integer.
fn error_name(api: &CudaApi, code: c_int) -> String {
    let name = unsafe { (api.error_name)(code) };
    if name.is_null() {
        return format!("unrecognised CUDA error {code}");
    }
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    // --- rung 1: parsing the process's own mapped files ---

    /// Representative `/proc/self/maps` content: a versioned runtime mapped
    /// several times (one line per segment), unrelated libraries, anonymous and
    /// special mappings, a path containing a space, and a deleted mapping.
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
                // Versioned sonames before the unversioned one: the pip CUDA
                // runtime wheels ship no unversioned symlink, which is the
                // original defect this ordering exists to prevent.
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

    /// Dangling but never dereferenced: `pin_all` only passes them to the API.
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
        // The success path's teardown, without a GPU: a real ring buffer, stubbed
        // CUDA. Registrations leaked here would be invisible from Python, so this
        // is checked through the injection point rather than externally.
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
        // Rollback across the real seam: the third array is rejected, so the two
        // that took must be unregistered and `Drop` must then do nothing more.
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

    /// The dev checkout's CUDA vendor package, so rung 2 is exercisable from
    /// `cargo test` (which has no Python interpreter to ask). See
    /// `docs/src/development.md` for installing the wheel.
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

    /// `cudaError_t cudaHostGetFlags(unsigned int*, void*)` — the runtime's own
    /// account of how an address is registered.
    type HostGetFlagsFn = unsafe extern "C" fn(*mut c_uint, *mut c_void) -> c_int;

    /// Resolve the runtime, or `None` on a machine without one. Rust has no
    /// native test skip, so these tests report and pass rather than fail on
    /// CPU-only CI.
    fn real_api() -> Option<&'static CudaApi> {
        match api(&dev_vendor_roots()) {
            Ok(api) => Some(api),
            Err(e) => {
                eprintln!("skipping: no CUDA runtime on this machine\n{e}");
                None
            }
        }
    }

    /// `cudaHostGetFlags`, looked up separately from the [`CudaApi`] under test.
    /// Confirming registration through echo's own function pointers would only
    /// prove echo believes it succeeded.
    ///
    /// Call only after resolution has run, so the runtime is mapped and this
    /// re-opens it (`RTLD_NOLOAD`) rather than loading a second copy.
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

        // Drop unregisters; the same address must no longer be known to the
        // runtime as registered host memory.
        drop(ring);
        let code = unsafe { (host_get_flags())(&mut flags, address as *mut c_void) };
        assert_ne!(
            code, CUDA_SUCCESS,
            "drop should have unregistered the ring buffer"
        );
    }

    #[test]
    fn pinning_twice_in_one_process_reuses_the_resolved_runtime() {
        // Two servers in one process must each pin with no extra configuration.
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
}
