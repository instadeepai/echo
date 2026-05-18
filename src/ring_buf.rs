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

pub struct PytreeRingBuf {
    /// One contiguous buffer per array in the flattened pytree.
    buffers: Vec<UnsafeCell<Vec<u8>>>,
    /// Bytes per slot for each array.
    slot_bytes: Vec<usize>,
    /// Total number of slots.
    capacity: usize,
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

        let buffers = slot_bytes
            .iter()
            .map(|&bytes| UnsafeCell::new(vec![0u8; bytes * capacity]))
            .collect();

        Self {
            buffers,
            slot_bytes,
            capacity,
        }
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
