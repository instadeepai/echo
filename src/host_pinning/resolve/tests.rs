//! Tests for the resolution ladder. No GPU and no CUDA install required: the
//! rungs are pure functions of the process's mappings and the filesystem.

use super::*;

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
