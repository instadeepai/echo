use echo::ring_buf::PytreeRingBuf;

#[test]
fn test_new_basic() {
    let buf = PytreeRingBuf::new(vec![16, 32], 12, 4);
    assert_eq!(buf.capacity(), 12);
    assert_eq!(buf.num_arrays(), 2);
}

#[test]
#[should_panic(expected = "capacity (5) must be a multiple of batch_size (3)")]
fn test_capacity_not_multiple_of_batch_size() {
    PytreeRingBuf::new(vec![8], 5, 3);
}

#[test]
#[should_panic(expected = "capacity must be > 0")]
fn test_capacity_zero() {
    PytreeRingBuf::new(vec![8], 0, 1);
}

#[test]
#[should_panic(expected = "slot_bytes must not be empty")]
fn test_empty_slot_bytes() {
    PytreeRingBuf::new(vec![], 4, 2);
}

#[test]
fn test_slot_mut_and_slot_ref() {
    let buf = PytreeRingBuf::new(vec![4], 8, 4);

    // Write to slot 2, array 0
    let data: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];
    unsafe {
        let ptr = buf.slot_mut(2, 0);
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, 4);
    }

    // Read back via slot_ref
    let view = buf.slot_ref(2, 0);
    assert_eq!(view, &[0xAA, 0xBB, 0xCC, 0xDD]);
}

#[test]
fn test_slots_are_non_overlapping() {
    let buf = PytreeRingBuf::new(vec![4], 8, 4);

    let data0: [u8; 4] = [0x11, 0x22, 0x33, 0x44];
    let data1: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];

    unsafe {
        let ptr0 = buf.slot_mut(0, 0);
        std::ptr::copy_nonoverlapping(data0.as_ptr(), ptr0, 4);

        let ptr1 = buf.slot_mut(1, 0);
        std::ptr::copy_nonoverlapping(data1.as_ptr(), ptr1, 4);
    }

    assert_eq!(buf.slot_ref(0, 0), &[0x11, 0x22, 0x33, 0x44]);
    assert_eq!(buf.slot_ref(1, 0), &[0xAA, 0xBB, 0xCC, 0xDD]);
}

#[test]
fn test_range_ptr_contiguous() {
    // slot_bytes = 4, capacity = 4, batch_size = 2
    let buf = PytreeRingBuf::new(vec![4], 4, 2);

    // Write to slot 0 and slot 1
    let data0: [u8; 4] = [0x01, 0x02, 0x03, 0x04];
    let data1: [u8; 4] = [0x05, 0x06, 0x07, 0x08];

    unsafe {
        let ptr0 = buf.slot_mut(0, 0);
        std::ptr::copy_nonoverlapping(data0.as_ptr(), ptr0, 4);

        let ptr1 = buf.slot_mut(1, 0);
        std::ptr::copy_nonoverlapping(data1.as_ptr(), ptr1, 4);
    }

    let (raw_ptr, total_len) = buf.range_ptr(0, 0, 2);
    assert_eq!(total_len, 8);

    let slice = unsafe { std::slice::from_raw_parts(raw_ptr as *const u8, total_len) };
    assert_eq!(slice, &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]);
}

#[test]
fn test_range_ptr_offset() {
    // slot_bytes = 4, capacity = 6, batch_size = 3
    let buf = PytreeRingBuf::new(vec![4], 6, 3);

    // Write distinct data to all 6 slots
    for slot in 0..6usize {
        let val = (slot as u8 + 1) * 0x11;
        let data = [val; 4];
        unsafe {
            let ptr = buf.slot_mut(slot, 0);
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, 4);
        }
    }

    // range_ptr starting at slot 3, count 3
    let (raw_ptr, total_len) = buf.range_ptr(0, 3, 3);
    assert_eq!(total_len, 12);

    let slice = unsafe { std::slice::from_raw_parts(raw_ptr as *const u8, total_len) };

    // slot 3 = 0x44, slot 4 = 0x55, slot 5 = 0x66, each repeated 4 times
    let expected: Vec<u8> = [0x44u8; 4]
        .iter()
        .chain([0x55u8; 4].iter())
        .chain([0x66u8; 4].iter())
        .copied()
        .collect();
    assert_eq!(slice, expected.as_slice());
}
