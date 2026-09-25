//! Raw ring buffer backing store for pytree samples.
//!
//! NOT thread-safe on its own. `PytreeRingBuf` uses `UnsafeCell` for
//! interior mutability and exposes unsynchronized read/write primitives
//! (`slot_mut`, `slot_ref`, `range_ptr`). Concurrent access is only sound
//! when the caller enforces disjointness — e.g. the `Store` partitions
//! slots so that no two writers touch the same slot, and the consumer
//! reads only after writers have committed.
//!
//! Do not use this type directly from multiple threads without external
//! coordination.

use std::cell::UnsafeCell;
use std::path::PathBuf;

use crate::host_pinning::{self, CudaApi, PinError, Region};

/// Where one array lives in the ring: its slot 0, and the stride between
/// consecutive slots. See [`PytreeRingBuf::array_layout`].
#[derive(Clone, Copy)]
pub struct ArrayLayout {
    base: *mut u8,
    slot_bytes: usize,
}

impl ArrayLayout {
    /// Pointer to `slot`'s data within this array.
    ///
    /// # Safety
    /// `slot` must be `< capacity` of the originating ring, and the caller must
    /// have exclusive access to this `(array, slot)` pair.
    #[inline(always)]
    pub unsafe fn slot(&self, slot: usize) -> *mut u8 {
        self.base.add(slot * self.slot_bytes)
    }

    /// Bytes per slot — also the stride between consecutive slots.
    #[inline(always)]
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }
}

pub struct PytreeRingBuf {
    /// One contiguous buffer per array in the flattened pytree.
    buffers: Vec<UnsafeCell<Vec<u8>>>,
    /// Base pointer and stride per array, resolved once in `new`.
    ///
    /// This is the write path's index, not `buffers`: deriving a slot address
    /// from `buffers` costs a bounds check and a load through the `UnsafeCell`
    /// on every access, and the compiler cannot hoist either, because writes
    /// through the resulting pointer may alias the `Vec` header it just read.
    ///
    /// Sound because the pointers address the heap allocations, not this
    /// struct: moving a `PytreeRingBuf` moves only the `Vec` headers, and the
    /// buffers are sized once in `new` and never resized.
    layouts: Vec<ArrayLayout>,
    /// Total number of slots.
    capacity: usize,
    /// The runtime the buffers are registered with, once they are. `Some` is
    /// `Drop`'s cue to unregister, and holding it here keeps `Drop` off a global.
    pinned_with: Option<CudaApi>,
}

impl PytreeRingBuf {
    /// Create a new ring buffer.
    /// Panics if capacity is zero, capacity % batch_size != 0, or slot_bytes is empty.
    pub fn new(slot_bytes: Vec<usize>, capacity: usize, batch_size: usize) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(
            capacity.is_multiple_of(batch_size),
            "capacity ({}) must be a multiple of batch_size ({})",
            capacity,
            batch_size
        );
        assert!(!slot_bytes.is_empty(), "slot_bytes must not be empty");

        let buffers: Vec<UnsafeCell<Vec<u8>>> = slot_bytes
            .iter()
            .map(|&bytes| UnsafeCell::new(vec![0u8; bytes * capacity]))
            .collect();

        let layouts = buffers
            .iter()
            .zip(&slot_bytes)
            .map(|(cell, &bytes)| ArrayLayout {
                // SAFETY: sole access — `buffers` is local and not yet shared.
                base: unsafe { (*cell.get()).as_mut_ptr() },
                slot_bytes: bytes,
            })
            .collect();

        Self {
            buffers,
            layouts,
            capacity,
            pinned_with: None,
        }
    }

    /// Page-lock every buffer, so a host-to-device copy of a sampled view is a
    /// DMA transfer rather than a chunked staging copy. All or nothing: a partial
    /// failure is rolled back before the error returns.
    ///
    /// Separate from `new` because a constructor returning `Err` never runs
    /// `Drop`, so registering there would need a hand-written unregister loop on
    /// the error path. Here `Drop` owns rollback and teardown alike.
    ///
    /// The buffers are contiguous and never reallocated, so a registration stays
    /// valid for the buffer's whole life. `cuda_vendor_roots` comes from Python's
    /// import machinery; see [`crate::host_pinning`].
    pub fn pin_host_memory(&mut self, cuda_vendor_roots: &[PathBuf]) -> Result<(), PinError> {
        self.pin_with(*host_pinning::api(cuda_vendor_roots)?)
    }

    /// [`Self::pin_host_memory`] against an already-resolved runtime. Split out
    /// so tests can drive registration and teardown with stubs, without a GPU.
    pub fn pin_with(&mut self, api: CudaApi) -> Result<(), PinError> {
        // SAFETY: the regions are `self`'s own allocations, never reallocated,
        // and `Drop` unregisters them before they are freed.
        unsafe { host_pinning::pin_all(&api, &self.regions())? };
        self.pinned_with = Some(api);
        Ok(())
    }

    /// Each backing buffer as a (pointer, length) pair for the CUDA runtime.
    fn regions(&self) -> Vec<Region> {
        self.buffers
            .iter()
            .map(|cell| {
                let buf = unsafe { &*cell.get() };
                Region {
                    ptr: buf.as_ptr() as *mut u8,
                    len: buf.len(),
                }
            })
            .collect()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    #[inline]
    pub fn num_arrays(&self) -> usize {
        self.layouts.len()
    }

    /// One array's base pointer and per-slot stride. A plain indexed load of a
    /// value resolved in `new` — cheap enough to call per slot, and cheaper
    /// still hoisted out of a loop over slots of the same array.
    ///
    /// # Safety
    /// - `array_idx` must be `< self.num_arrays()`
    /// - The layout points into `self`'s allocation. It stays valid for as long
    ///   as the ring lives, but the caller carries the same disjointness
    ///   obligation as [`Self::slot_mut`] for every slot it writes through.
    #[inline]
    pub unsafe fn array_layout(&self, array_idx: usize) -> ArrayLayout {
        debug_assert!(
            array_idx < self.layouts.len(),
            "array_idx {array_idx} out of bounds"
        );
        *self.layouts.get_unchecked(array_idx)
    }

    /// Mutable pointer to a slot's array data for writing.
    /// # Safety
    /// - `slot` must be `< self.capacity`
    /// - `array_idx` must be `< self.num_arrays()`
    /// - Caller must ensure exclusive access to this slot index; no other thread
    ///   may read or write the same `(array_idx, slot)` pair concurrently.
    ///
    /// Different slot indices access non-overlapping memory regions.
    #[inline]
    pub unsafe fn slot_mut(&self, slot: usize, array_idx: usize) -> *mut u8 {
        debug_assert!(
            slot < self.capacity,
            "slot {slot} out of bounds (capacity {})",
            self.capacity
        );
        self.array_layout(array_idx).slot(slot)
    }

    /// Immutable view into a slot's array data.
    pub fn slot_ref(&self, slot: usize, array_idx: usize) -> &[u8] {
        debug_assert!(
            slot < self.capacity,
            "slot {slot} out of bounds (capacity {})",
            self.capacity
        );
        // Indexing `layouts` keeps the array bound checked on this safe path.
        let bytes = self.layouts[array_idx].slot_bytes();
        let offset = slot * bytes;
        let buf = unsafe { &*self.buffers[array_idx].get() };
        &buf[offset..offset + bytes]
    }

    /// Raw pointer + byte length for a contiguous range of slots in one array.
    /// Used to construct zero-copy ConsumerView.
    /// start + count must not wrap (guaranteed when capacity % batch_size == 0).
    pub fn range_ptr(&self, array_idx: usize, start: usize, count: usize) -> (usize, usize) {
        debug_assert!(
            start + count <= self.capacity,
            "range must not wrap: start={start} count={count} capacity={}",
            self.capacity
        );
        // Once per array per batch — the bound check stays, this is not hot.
        let layout = self.layouts[array_idx];
        let ptr = layout.base.wrapping_add(start * layout.slot_bytes());
        (ptr as usize, count * layout.slot_bytes())
    }
}

// Safety: PytreeRingBuf uses UnsafeCell for interior mutability.
// The Store guarantees that concurrent writers access disjoint slots,
// and the consumer only reads after writers have finished.
unsafe impl Sync for PytreeRingBuf {}

// Safety: All data is heap-allocated and owned; transfer between threads is safe.
unsafe impl Send for PytreeRingBuf {}

impl Drop for PytreeRingBuf {
    fn drop(&mut self) {
        let Some(api) = self.pinned_with else {
            return;
        };
        // Drop::drop runs before the fields are dropped, so the memory is still
        // valid here.
        // SAFETY: `pinned_with` is Some only after `pin_with` registered exactly
        // these regions through this same api.
        unsafe { host_pinning::unpin_all(&api, &self.regions()) };
    }
}
