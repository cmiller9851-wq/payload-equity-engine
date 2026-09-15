use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum RingBufferError {
    #[error("Buffer is full - apply backpressure")]
    Full,
    #[error("Buffer is empty")]
    Empty,
}

/// A zero-allocation, lock-free, Single-Producer Single-Consumer (SPSC) ring buffer.
///
/// Guarantees deterministic O(1) execution time and compile-time thread safety
/// using CPU atomic memory barriers and const generics.
pub struct SpscRingBuffer<T, const N: usize> {
    buffer: [Option<T>; N],
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T, const N: usize> SpscRingBuffer<T, N> {
    const INIT: Option<T> = None;

    /// Creates a statically allocated ring buffer.
    /// Fails at compile-time/initialization if capacity N is not a power of two.
    pub fn new() -> Self {
        assert!(
            N > 0 && (N & (N - 1)) == 0,
            "Buffer capacity N must be a non-zero power of two"
        );
        Self {
            buffer: [Self::INIT; N],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Pushes an item into the buffer.
    /// Non-blocking, lock-free, and safe under concurrent single-producer usage.
    pub fn push(&mut self, item: T) -> Result<(), RingBufferError> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        // Check if full using wrapping arithmetic to prevent overflow bugs
        if head.wrapping_sub(tail) >= N {
            return Err(RingBufferError::Full);
        }

        // Mask index using bitwise AND (valid because N is guaranteed power of 2)
        let index = head & (N - 1);
        self.buffer[index] = Some(item);

        // Release barrier ensures payload write is visible before head updates
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Pops an item from the buffer.
    /// Non-blocking, lock-free, and safe under concurrent single-consumer usage.
    pub fn pop(&mut self) -> Result<T, RingBufferError> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return Err(RingBufferError::Empty);
        }

        let index = tail & (N - 1);
        let item = self
            .buffer[index]
            .take()
            .expect("Verified slot state non-empty");

        // Release barrier ensures slot clearing completes before tail updates
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Ok(item)
    }

    /// Returns the exact number of unconsumed items in the buffer.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
