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

pub struct PytreeRingBuf {
    /// One contiguous buffer per array in the flattened pytree.
    buffers: Vec<UnsafeCell<Vec<u8>>>,
    /// Bytes per slot for each array.
    slot_bytes: Vec<usize>,
    /// Total number of slots.
    capacity: usize,
    /// The CUDA runtime the buffers are registered with, once they are.
    /// `Some` is exactly the condition for `Drop` having registrations to
    /// reverse, and holding it here keeps `Drop` off any process-global.
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

        Self {
            buffers,
            slot_bytes,
            capacity,
            pinned_with: None,
        }
    }

    /// Page-lock every buffer, so a downstream host-to-device copy of a sampled
    /// view is a DMA transfer rather than a chunked staging copy.
    ///
    /// Either every buffer ends up locked or none does — a partial failure is
    /// rolled back before the error returns. That is why this is a separate
    /// fallible step on a constructed buffer rather than part of `new`: a
    /// constructor returning `Err` never runs `Drop`, so registering there would
    /// need a hand-written unregister loop on the error path. Here `Drop` owns
    /// rollback for both the failure path and normal teardown.
    ///
    /// `cuda_vendor_roots` are the CUDA vendor package directories located
    /// through Python's import machinery; see [`crate::host_pinning`].
    ///
    /// The buffers are contiguous and never reallocated, so a registration stays
    /// valid for the buffer's whole life.
    pub(crate) fn pin_host_memory(
        &mut self,
        cuda_vendor_roots: &[PathBuf],
    ) -> Result<(), PinError> {
        self.pin_with(*host_pinning::api(cuda_vendor_roots)?)
    }

    /// [`Self::pin_host_memory`] against an already-resolved runtime.
    ///
    /// Split out so tests can drive registration and teardown with stubbed CUDA
    /// entry points, on a machine with no GPU.
    pub(crate) fn pin_with(&mut self, api: CudaApi) -> Result<(), PinError> {
        // SAFETY: the regions are this buffer's own allocations, which live as
        // long as `self` and are never reallocated, and `Drop` unregisters them
        // before they are freed.
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

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn num_arrays(&self) -> usize {
        self.slot_bytes.len()
    }

    /// Mutable pointer to a slot's array data for writing.
    /// # Safety
    /// - `slot` must be `< self.capacity`
    /// - `array_idx` must be `< self.num_arrays()`
    /// - Caller must ensure exclusive access to this slot index; no other thread
    ///   may read or write the same `(array_idx, slot)` pair concurrently.
    ///
    /// Different slot indices access non-overlapping memory regions.
    pub unsafe fn slot_mut(&self, slot: usize, array_idx: usize) -> *mut u8 {
        debug_assert!(
            slot < self.capacity,
            "slot {slot} out of bounds (capacity {})",
            self.capacity
        );
        debug_assert!(
            array_idx < self.slot_bytes.len(),
            "array_idx {array_idx} out of bounds"
        );
        let offset = slot * self.slot_bytes[array_idx];
        (*self.buffers[array_idx].get()).as_mut_ptr().add(offset)
    }

    /// Immutable view into a slot's array data.
    pub fn slot_ref(&self, slot: usize, array_idx: usize) -> &[u8] {
        debug_assert!(
            slot < self.capacity,
            "slot {slot} out of bounds (capacity {})",
            self.capacity
        );
        debug_assert!(
            array_idx < self.slot_bytes.len(),
            "array_idx {array_idx} out of bounds"
        );
        let bytes = self.slot_bytes[array_idx];
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
        let bytes = self.slot_bytes[array_idx];
        let offset = start * bytes;
        let buf = unsafe { &*self.buffers[array_idx].get() };
        let ptr = buf.as_ptr().wrapping_add(offset);
        (ptr as usize, count * bytes)
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
        // Unregister before the backing Vecs are freed. Drop::drop runs before
        // the struct's fields are dropped, so the memory is still valid.
        // SAFETY: `pinned_with` is Some only after `pin_with` registered exactly
        // these regions through this same api, and they are still valid here.
        unsafe { host_pinning::unpin_all(&api, &self.regions()) };
    }
}
