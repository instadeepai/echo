//! Finding the CUDA runtime: a three-rung ladder, tried in order, accumulating
//! every attempted path so a total failure can say what it tried.

use std::ffi::{CStr, CString};
use std::os::raw::c_void;
use std::path::PathBuf;
use std::sync::OnceLock;

use super::{CudaApi, ErrorNameFn, FreeFn, PinError, RegisterFn, UnregisterFn};

/// Current and previous CUDA major versions, then the unversioned soname.
///
/// The unversioned name is last and rarely resolves: the pip CUDA runtime
/// wheels ship only the versioned soname, with no symlink and no ldconfig entry.
const SONAMES: [&str; 3] = ["libcudart.so.13", "libcudart.so.12", "libcudart.so"];

/// Which rung of the resolution ladder produced a candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rung {
    /// Rung 1: a runtime the process already has mapped.
    AlreadyLoaded,
    /// Rung 2: a runtime shipped by an installed CUDA wheel.
    InstalledWheel,
    /// Rung 3: a soname, left to the system loader.
    Soname,
}

impl Rung {
    pub fn label(self) -> &'static str {
        match self {
            Rung::AlreadyLoaded => "already-loaded scan",
            Rung::InstalledWheel => "installed-wheel search",
            Rung::Soname => "soname load",
        }
    }
}

/// One thing to hand to `dlopen`, in ladder order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    /// An absolute path (rungs 1 and 2) or a bare soname (rung 3).
    pub name: String,
    pub rung: Rung,
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
pub fn scan_mapped_runtimes(maps: &str) -> Vec<String> {
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
pub fn search_wheel_roots(roots: &[PathBuf]) -> Vec<String> {
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
pub fn candidates(maps: &str, vendor_roots: &[PathBuf]) -> Vec<Candidate> {
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
        Ok(CudaApi::new(
            std::mem::transmute::<*mut c_void, RegisterFn>(symbol(handle, c"cudaHostRegister")?),
            std::mem::transmute::<*mut c_void, UnregisterFn>(symbol(
                handle,
                c"cudaHostUnregister",
            )?),
            std::mem::transmute::<*mut c_void, ErrorNameFn>(symbol(handle, c"cudaGetErrorName")?),
            std::mem::transmute::<*mut c_void, FreeFn>(symbol(handle, c"cudaFree")?),
        ))
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
pub fn api(vendor_roots: &[PathBuf]) -> Result<&'static CudaApi, PinError> {
    if let Some(api) = API.get() {
        return Ok(api);
    }
    let resolved = resolve(vendor_roots)?;
    // Two threads racing here both resolve; `dlopen` is idempotent and
    // reference-counted, so the loser just drops an identical set of pointers.
    Ok(API.get_or_init(|| resolved))
}
