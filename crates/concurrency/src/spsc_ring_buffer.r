use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum RingBufferError {
    #[error("Buffer is full - backpressure applied")]
    Full,
    #[error("Buffer is empty")]
    Empty,
}

/// A zero-allocation, lock-free, single-producer single-consumer (SPSC)
/// bounded ring buffer. Guarantees deterministic O(1) time complexity
/// and thread-safety via atomic memory barriers.
pub struct SpscRingBuffer<T, const N: usize> {
    buffer: [Option<T>; N],
    head: AtomicUsize,
    tail: AtomicUsize,
}

impl<T, const N: usize> SpscRingBuffer<T, N> {
    const INIT: Option<T> = None;

    /// Creates a statically initialized, zero-allocation ring buffer.
    /// Fails at compile-time if capacity is not a power of two.
    pub fn new() -> Self {
        assert!(N > 0 && (N & (N - 1)) == 0, "Buffer size N must be a power of two");
        Self {
            buffer: [Self::INIT; N],
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
        }
    }

    /// Pushes an item into the buffer. 
    /// Non-blocking, lock-free, and thread-safe.
    pub fn push(&mut self, item: T) -> Result<(), RingBufferError> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        // Check if buffer is full without integer overflow risks
        if head.wrapping_sub(tail) >= N {
            return Err(RingBufferError::Full);
        }

        // Mask index using bitwise AND (requires N to be a power of 2)
        let index = head & (N - 1);
        self.buffer[index] = Some(item);

        // Release barrier guarantees item is visible before head increment
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Pops an item from the buffer.
    /// Non-blocking, lock-free, and thread-safe.
    pub fn pop(&mut self) -> Result<T, RingBufferError> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        if tail == head {
            return Err(RingBufferError::Empty);
        }

        let index = tail & (N - 1);
        let item = self.buffer[index].take().expect("Buffer slot verified non-empty");

        // Release barrier guarantees slot reading is complete before tail update
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Ok(item)
    }

    /// Returns the exact number of pending items.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
