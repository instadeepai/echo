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

/// `cudaHostRegisterPortable`: valid in every CUDA context in the process,
/// including ones created later. This is why pinning takes no device argument
/// and why N servers across N GPUs in one process need no configuration.
const CUDA_HOST_REGISTER_PORTABLE: c_uint = 0x01;

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

// ---------------------------------------------------------------------------
// Resolving the CUDA runtime
// ---------------------------------------------------------------------------

/// Current and previous CUDA major versions, then the unversioned soname.
///
/// The unversioned name is last and rarely resolves: the pip CUDA runtime
/// wheels ship only the versioned soname, with no symlink and no ldconfig entry.
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

/// Rung 1: runtimes the process already has mapped, in first-seen order.
///
/// Hits whenever the framework has already initialised CUDA. `dlopen` on the
/// absolute path of a mapped library reuses that mapping rather than loading a
/// second copy.
fn scan_mapped_runtimes(maps: &str) -> Vec<String> {
    let mut found: Vec<String> = Vec::new();
    for line in maps.lines() {
        // The pathname is the last field and is absolute, so it starts at the
        // first " /". Located this way, not by splitting on whitespace, because
        // paths may contain spaces. Unlinked files get a " (deleted)" suffix.
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

/// `nvidia/cu13/lib/libcudart.so.13` is three deep; one spare for a relayout.
const WHEEL_SEARCH_DEPTH: usize = 4;

/// Rung 2: runtimes shipped by installed wheels, newest major first. Lets the
/// server be constructed before the framework has loaded CUDA.
///
/// `roots` are the CUDA *vendor* package directories, found by the caller
/// through Python's import machinery. Searching the whole vendor package rather
/// than a named component is load-bearing: CUDA 13 ships one consolidated wheel
/// (`nvidia/cu13/lib/`), CUDA 12 one per component (`nvidia/cuda_runtime/lib/`),
/// so naming the component finds nothing on a CUDA 13 install.
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

/// The full ladder, in the order it will be tried. Pure, so the ordering is
/// unit-testable without a GPU: reordering it is how this feature broke before.
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
/// The handle is never `dlclose`d: the registrations it backs must outlive it.
fn open(candidate: &Candidate) -> Result<CudaApi, String> {
    let name = CString::new(candidate.name.as_str())
        .map_err(|_| "path contains an interior NUL byte".to_string())?;

    // RTLD_LOCAL, so this does not change how any other library in the process
    // resolves its symbols.
    let flags = libc::RTLD_NOW | libc::RTLD_LOCAL;
    let handle = unsafe {
        // Rung 3 prefers a copy the process already has, under whatever path,
        // over pulling a second one in from the loader path.
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
/// where it looked.
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

/// The CUDA runtime, resolving it on first use. Only called when pinning was
/// asked for, so the default-off path loads nothing.
///
/// Failures are not cached: a process that retries after its framework has
/// initialised CUDA should get the later, better answer.
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
pub(crate) unsafe fn pin_all(api: &CudaApi, regions: &[Region]) -> Result<(), PinError> {
    // Registration against an uninitialised runtime fails, so force init.
    // Freeing null frees nothing and never changes the current device, but it is
    // not free: it creates the primary context on the current device, costing
    // that context's memory (~100 MB). Constructing after the framework has
    // initialised CUDA — the documented order — means it already exists.
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

#[cfg(test)]
mod tests;
