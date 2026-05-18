use echo::selector::{FifoRemover, FifoSampler, Remover, SampleResult, Sampler};

// === FifoSampler ===

#[test]
fn test_sampler_commit_and_select() {
    let sampler = FifoSampler::new(3, 6);
    sampler.commit(0);
    sampler.commit(1);
    sampler.commit(2);

    let result = sampler.select(3).unwrap();
    assert_eq!(result, SampleResult::Contiguous { start: 0, count: 3 });
}

#[test]
fn test_sampler_try_select_not_enough() {
    let sampler = FifoSampler::new(3, 6);
    sampler.commit(0);
    sampler.commit(1);
    assert!(sampler.try_select(3).is_none());
}

#[test]
fn test_sampler_try_select_enough() {
    let sampler = FifoSampler::new(3, 6);
    sampler.commit(0);
    sampler.commit(1);
    sampler.commit(2);
    let result = sampler.try_select(3).unwrap();
    assert_eq!(result, SampleResult::Contiguous { start: 0, count: 3 });
}

#[test]
fn test_sampler_advances_head() {
    let sampler = FifoSampler::new(2, 6);
    for i in 0..4 {
        sampler.commit(i);
    }
    let r1 = sampler.select(2).unwrap();
    assert_eq!(r1, SampleResult::Contiguous { start: 0, count: 2 });
    let r2 = sampler.select(2).unwrap();
    assert_eq!(r2, SampleResult::Contiguous { start: 2, count: 2 });
}

#[test]
fn test_sampler_wrap_around() {
    let sampler = FifoSampler::new(3, 6);
    for i in 0..6 {
        sampler.commit(i);
    }
    sampler.select(3).unwrap(); // head = 3
    sampler.select(3).unwrap(); // head = 6 → wraps to 0
    for i in 6..9 {
        sampler.commit(i);
    }
    let r = sampler.select(3).unwrap();
    assert_eq!(r, SampleResult::Contiguous { start: 0, count: 3 });
}

#[test]
fn test_sampler_shutdown_unblocks() {
    let sampler = std::sync::Arc::new(FifoSampler::new(3, 6));
    let s = sampler.clone();
    let handle = std::thread::spawn(move || s.select(3));
    std::thread::sleep(std::time::Duration::from_millis(50));
    sampler.shutdown();
    assert!(handle.join().unwrap().is_none());
}

#[test]
fn test_sampler_out_of_order_writers() {
    // Simulate the race: 3 writers claim slots 0,1,2 but commit out of order.
    let sampler = std::sync::Arc::new(FifoSampler::new(3, 6));
    let s0 = sampler.clone();
    let s1 = sampler.clone();
    let s2 = sampler.clone();

    // Writer 2 and 1 commit first, writer 0 commits last.
    let h2 = std::thread::spawn(move || s2.commit(2));
    let h1 = std::thread::spawn(move || s1.commit(1));
    // Small delay so writers 1 and 2 are spinning.
    std::thread::sleep(std::time::Duration::from_millis(10));
    let h0 = std::thread::spawn(move || s0.commit(0));

    h0.join().unwrap();
    h1.join().unwrap();
    h2.join().unwrap();

    // All three committed in order; batch should be ready.
    let result = sampler.try_select(3).unwrap();
    assert_eq!(result, SampleResult::Contiguous { start: 0, count: 3 });
}

// === FifoRemover ===

#[test]
fn test_remover_advance_and_read_pos() {
    let remover = FifoRemover::new();
    assert_eq!(remover.read_pos(), 0);
    remover.remove(3);
    assert_eq!(remover.read_pos(), 3);
    remover.remove(3);
    assert_eq!(remover.read_pos(), 6);
}

// === FifoSampler::queue_size ===

#[test]
fn test_fifo_sampler_queue_size_empty() {
    let s = FifoSampler::new(2, 8);
    assert_eq!(s.queue_size(), 0);
}

#[test]
fn test_fifo_sampler_queue_size_after_commits() {
    let s = FifoSampler::new(2, 8);
    s.commit(0);
    s.commit(1);
    s.commit(2);
    assert_eq!(s.queue_size(), 3, "3 items committed, none sampled");
}

#[test]
fn test_fifo_sampler_queue_size_after_select() {
    let s = FifoSampler::new(2, 8);
    s.commit(0);
    s.commit(1);
    let _ = s.try_select(2).expect("should have a batch");
    assert_eq!(s.queue_size(), 0, "2 committed, 2 sampled -> 0 ready");
}
