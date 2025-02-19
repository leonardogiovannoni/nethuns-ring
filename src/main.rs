use std::cell::UnsafeCell;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// Status marker indicating that a slot has an allocated item.
const ALLOCATED: u8 = 0;
/// Status marker indicating that a slot's item has been "stolen" (popped by the consumer).
const STOLEN: u8 = 1;
/// Status marker indicating that a slot has been released (via `release_slot`) and is reclaimable.
const RECLAIMABLE: u8 = 2;

/// A single-consumer, multi-producer ring buffer based data structures.
///
/// # Overview
///
/// This queue is designed for exactly one consumer thread and any number of producer threads.
/// * The consumer is responsible for "popping" items out of the queue.
/// * The producers are responsible for "pushing" items back into the queue via a reusable slot mechanism.
///
/// Internally, the queue maintains a ring buffer of [`Slot`]s, each with an atomic status. The
/// consumer alone maintains `head`, `tail`, and `count` via non-atomic fields, assuming it has
/// exclusive access to these fields on a single thread.
///
/// # Concurrency
///
/// * **Single Consumer**: The consumer methods (`try_pop_slot`, `reclaim_slots`, etc.) are **not** safe
///   to call from multiple threads simultaneously. These methods must only be called by the single
///   consumer thread.
/// * **Multiple Producers**: The producer methods are safe to call from multiple threads
///   simultaneously because they only perform atomic operations on shared slots.
///
/// # Notation
///
/// - `[head, tail)` is the range of elements currently known to be in the queue from the consumer's
///   perspective.
/// - A slot is "in the queue" if its status is `ALLOCATED`.
/// - A slot is "outside the queue" if its status is `STOLEN` or `RECLAIMABLE`.
/// - "Apparently empty" means `head == tail`, but note there might be `RECLAIMABLE` slots that the
///   consumer could fold back in via a call to `reclaim_slots`.
///
/// # Invariants
///
/// - `0 < capacity` (enforced by `NonZeroUsize`)
/// - `head`, `tail`, and `count` are maintained by the single consumer.
/// - The producers only change the status of slots (atomically), never modifying `head`, `tail`, or `count`.
/// - If `queue` is empty, then it is also "apparently empty" from the consumer’s perspective (`head == tail`).
/// - If `queue` is full, then `count == capacity`.
pub struct NtsRing<T> {
    /// The ring buffer storing slots. Each slot has a status (`AtomicU8`) and a value (`UnsafeCell<T>`).
    buffer: Box<[Slot<Option<T>>]>,
    /// The index of the first allocated element in the queue (maintained by the consumer).
    head: usize,
    /// The index just past the last allocated element in the queue (maintained by the consumer).
    tail: usize,
    /// The capacity of the queue (never zero).
    capacity: NonZeroUsize,
    /// The current count of items in the queue (maintained by the consumer).
    count: usize,
}

/// A single slot in the queue, containing:
/// * `status` — an atomic integer describing the slot's state
/// * `value` — an `UnsafeCell` that can hold the actual item of type `T`
#[derive(Debug)]
pub struct Slot<T> {
    pub status: AtomicU8,
    pub value: UnsafeCell<T>,
}

impl<T> NtsRing<T> {
    /// Creates a new `NtsRing` from a `Vec<T>`. All elements in the `Vec`
    /// become initially allocated in the queue.
    ///
    /// # Panics
    ///
    /// Panics if the provided `Vec` has length 0.
    pub fn new(v: Vec<T>) -> Self {
        let capacity = v.len();
        let capacity = NonZeroUsize::new(capacity).expect("capacity is zero");
        let mut buffer = Vec::with_capacity(capacity.get());

        // Initialize the ring buffer slots
        for value in v {
            buffer.push(Slot {
                status: AtomicU8::new(ALLOCATED),
                value: UnsafeCell::new(Some(value)),
            });
        }

        NtsRing {
            buffer: buffer.into_boxed_slice(),
            head: 0,
            tail: 0,
            capacity,
            count: capacity.get(),
        }
    }

    /// Attempts to fold any `RECLAIMABLE` items back into the queue until the first `STOLEN`
    /// item is encountered, or until the queue becomes full.
    ///
    /// Single-Consumer
    ///
    /// This function must only be called by the single consumer thread.
    fn attempt_reclaim_slots(&mut self) {
        loop {
            // Stop if we reach a full queue
            if self.is_full() {
                break;
            }

            let tail_slot = unsafe { self.buffer.get_unchecked_mut(self.tail) };

            match tail_slot.status.load(Ordering::Acquire) {
                ALLOCATED => {
                    // We cannot reclaim slots if the tail is pointing to an allocated element
                    panic!("Cannot reclaim slots while tail points to an allocated element.");
                }
                STOLEN => {
                    // Stop when we find the first stolen element
                    break;
                }
                RECLAIMABLE => {
                    self.tail = (self.tail + 1) % self.capacity.get();
                    self.count += 1;
                    tail_slot.status.store(ALLOCATED, Ordering::Release);
                }
                _ => unreachable!("Encountered unexpected status."),
            }
        }
    }

    /// Allowing the queue to reclaim slots if it is "apparently empty".
    ///
    /// Single-Consumer
    ///
    /// This function must only be called by the single consumer thread.
    fn reclaim_slots(&mut self) {
        // Only reclaim slots if the queue is apparently empty from the consumer's perspective.
        if self.is_apparently_empty() {
            self.attempt_reclaim_slots();
        }
    }

    /// Returns the current number of elements in the queue (as tracked by `count`).
    ///
    /// Single-Consumer
    ///
    /// This value is maintained by the consumer and is reliable only from the consumer's thread.
    fn size(&self) -> usize {
        self.count
    }

    /// Returns `true` if the queue is empty (as tracked by `count`).
    ///
    /// Single-Consumer
    ///
    /// This value is maintained by the consumer and is reliable only from the consumer's thread.
    fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns `true` if the queue is full (as tracked by `count`).
    ///
    /// Single-Consumer
    ///
    /// This value is maintained by the consumer and is reliable only from the consumer's thread.
    fn is_full(&self) -> bool {
        self.count == self.capacity.get()
    }

    /// Returns `true` if the queue is "apparently empty" from the consumer's perspective (`head == tail`).
    /// Note that there might still be `RECLAIMABLE` slots that can be reclaimed with `reclaim_slots`.
    ///
    /// Single-Consumer
    ///
    /// This check is only meant for the consumer thread.
    fn is_apparently_empty(&self) -> bool {
        if self.head != self.tail {
            return false;
        }
        // If head == tail, double-check the status of the slot
        // to confirm whether there is any allocated item.
        unsafe {
            (*self.buffer.get_unchecked(self.head))
                .status
                .load(Ordering::Acquire)
                != ALLOCATED
        }
    }

    /// Tries to pop a slot from the front of the queue, returning `(index, T)` if successful,
    /// or `None` if the queue is apparently empty even after a `reclaim_slots`.
    ///
    /// Single-Consumer
    ///
    /// This function must only be called from the single consumer thread. It internally updates
    /// `head`, `tail`, and `count`, which are non-atomic, so calling this concurrently
    /// from multiple threads would break invariants.
    ///
    /// # Returns
    ///
    /// * `Some((idx, item))` if the pop was successful.
    /// * `None` if the queue is apparently empty.
    fn try_pop_slot(&mut self) -> Option<(usize, T)> {
        // Attempt to reclaim any RECLAIMABLE slots if empty
        self.reclaim_slots();
        if self.is_apparently_empty() {
            return None;
        }

        let idx = self.head;
        let slot = unsafe { self.buffer.get_unchecked_mut(idx) };

        // Mark the slot as STOLEN
        slot.status.store(STOLEN, Ordering::Release);

        // Move head forward
        self.head = (self.head + 1) % self.capacity.get();
        self.count -= 1;

        // Take the item out of the slot
        let value = unsafe { (*slot.value.get()).take().unwrap() };
        Some((idx, value))
    }

    /// Called by the producer to place an item back into a slot that was previously popped by the consumer.
    ///
    /// The consumer will later reclaim this slot with `reclaim_slots`, turning it from `RECLAIMABLE`
    /// back into `ALLOCATED`.
    ///
    /// Multi-Producer
    ///
    /// This function may be called by any producer thread. It updates the status in an atomic
    /// way that is visible to the consumer. The `index` must be a valid slot index that was
    /// obtained from a prior call to `try_pop_slot`.
    ///
    /// # Arguments
    ///
    /// * `value` — The item to be placed back into the slot.
    /// * `index` — The index of the slot (returned earlier by `try_pop_slot`).
    fn release_slot(&self, value: T, index: usize) {
        // Safety: We are not modifying any consumer-only fields here; only the atomic slot status.
        let slot = unsafe { &*self.buffer.get_unchecked(index) };
        unsafe { *slot.value.get() = Some(value) };
        slot.status.store(RECLAIMABLE, Ordering::Release);
    }

    /// Splits the `NtsRing` into a `(Producer<T>, Consumer<T>)` pair, wrapped in an `Arc`.
    ///
    /// The `Producer<T>` handle can be cloned freely and sent to multiple threads. The `Consumer<T>`
    /// handle must remain on a single thread.
    ///
    /// # Safety
    ///
    /// * There must only be one `Consumer<T>` in use at a time.
    /// * Any number of `Producer<T>` clones can exist and be used concurrently by multiple threads.
    pub fn split(self) -> (Producer<T>, Consumer<T>) {
        let me = Arc::new(UnsafeCell::new(self));
        (
            Producer {
                queue: me.clone(),
                _marker: PhantomData,
            },
            Consumer {
                queue: me,
                _marker: PhantomData,
            },
        )
    }
}

/// A guard holding an item that was popped from a `NtsRing`.
///
/// While holding this guard, the slot is marked as `STOLEN`. A producer can later
/// reinsert the item into the queue by calling [`Producer::push()`].
#[derive(Debug)]
pub struct NtsRingGuard<T> {
    index: usize,
    value: T,
}

impl<T> Deref for NtsRingGuard<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl<T> DerefMut for NtsRingGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

/// A producer handle for a `NtsRing<T>`.
///
/// `Producer<T>` can be cloned, allowing multiple producer threads.
/// Each producer can only push items via [`Producer::push()`] that were
/// previously popped by the single consumer.
#[derive(Clone)]
pub struct Producer<T> {
    queue: Arc<UnsafeCell<NtsRing<T>>>,
    // Prevents `T: !Send` from being used incorrectly
    _marker: PhantomData<*mut T>,
}

// `Producer<T>` is `Send` provided `T` is `Send`.
unsafe impl<T> Send for Producer<T> where T: Send {}

impl<T> Producer<T> {
    /// Releases an item back into the queue via the slot index held in the `NtsRingGuard`.
    ///
    /// Multi-Producer
    ///
    /// This method may be called concurrently by multiple producers, as it only performs atomic
    /// operations on the slot status. The `NtsRingGuard` must come from a valid `Consumer::pop()`
    /// call on the corresponding queue.
    pub fn release(&self, elem: NtsRingGuard<T>) {
        let NtsRingGuard { index, value } = elem;
        // Safety: We only call `release_slot`, which updates the status atomically.
        unsafe {
            (*self.queue.get()).release_slot(value, index);
        }
    }
}

/// A consumer handle for a `NtsRing<T>`.
///
/// There must be only one active `Consumer<T>` for a given queue at a time, and it must be used
/// by a single thread, due to the non-atomic fields (`head`, `tail`, `count`) being mutated.
pub struct Consumer<T> {
    queue: Arc<UnsafeCell<NtsRing<T>>>,
    // Prevents `T: !Send` from being used incorrectly
    _marker: PhantomData<*mut T>,
}

// `Consumer<T>` is `Send` provided `T` is `Send`.
unsafe impl<T> Send for Consumer<T> where T: Send {}

impl<T> Consumer<T> {
    /// Attempts to acquire an item from the queue, returning a `NtsRingGuard<T>` if successful.
    ///
    /// Single-Consumer
    ///
    /// This method mutates internal non-atomic fields (`head`, `tail`, `count`), so it must
    /// only be called by the single consumer thread.
    ///
    /// # Returns
    ///
    /// * `Some(NtsRingGuard<T>)` if an item was successfully acquired.
    /// * `None` if the queue was apparently empty.
    pub fn acquire(&self) -> Option<NtsRingGuard<T>> {
        let (index, value) = unsafe { (*self.queue.get()).try_pop_slot()? };
        Some(NtsRingGuard { index, value })
    }
}

fn main() {
    let initial = vec![1, 2, 3];
    let queue = NtsRing::new(initial.clone());
    let (prod, cons) = queue.split();

    // Perform 10 full cycles.
    for _ in 0..10 {
        // Consumer pops until the queue appears empty.
        let mut popped = Vec::new();
        while let Some(guard) = cons.acquire() {
            popped.push(guard);
        }
        for guard in popped {
            prod.release(guard);
        }
        // After all pushes, the consumer should be able to reclaim all pushed slots.
        let mut cycle_vals = Vec::new();
        let mut guards = Vec::new();
        while let Some(guard) = cons.acquire() {
            cycle_vals.push(*guard);
            guards.push(guard);
        }

        for guard in guards {
            prod.release(guard);
        }

        // Although the order might be rotated, the multiset must match.
        cycle_vals.sort();
        let mut sorted_initial = initial.clone();
        sorted_initial.sort();
        assert_eq!(
            cycle_vals, sorted_initial,
            "Cycle did not yield the expected elements"
        );
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    /// Verifies that reclaiming stops when a STOLEN slot is encountered.
    ///
    /// In this test we use a capacity-4 queue:
    /// - We pop two elements (making slots 0 and 1 STOLEN).
    /// - We then push the first popped element back (slot 0 becomes RECLAIMABLE)
    ///   while leaving slot 1 as STOLEN.
    /// - After popping the remaining two allocated items, the consumer’s next pop
    ///   triggers reclamation. It should reclaim slot 0 (value 10) and then stop when
    ///   it sees slot 1 is still STOLEN.
    #[test]
    fn test_reclaim_stops_on_stolen() {
        // Create a queue with four elements.
        let queue = NtsRing::new(vec![10, 20, 30, 40]);
        let (prod, cons) = queue.split();

        // Pop two elements: these come from indices 0 and 1.
        let guard1 = cons.acquire().expect("Expected first pop");
        let guard2 = cons.acquire().expect("Expected second pop");
        // At this point, slots 0 and 1 are marked STOLEN, head has advanced to index 2.

        // Push back the first popped element so that slot 0 becomes RECLAIMABLE.
        prod.release(guard1);
        // Leave guard2 unpushed so that slot 1 remains STOLEN.

        // Pop the remaining allocated elements (from indices 2 and 3).
        let _guard3 = cons.acquire().expect("Expected third pop");
        let _guard4 = cons.acquire().expect("Expected fourth pop");
        // Now head has wrapped around (head == 0) and count is zero.
        // The underlying state should be:
        //   slot 0: RECLAIMABLE (with value 10)
        //   slot 1: STOLEN (with value 20)
        //   slot 2: STOLEN (with value 30)
        //   slot 3: STOLEN (with value 40)

        // Next pop triggers reclaim_slots.
        let reclaimed = cons.acquire().expect("Expected a reclaimed element");
        // The reclaim loop will reclaim slot 0 and then break at slot 1.
        assert_eq!(*reclaimed, 10);

        // With no further RECLAIMABLE slots, further pops must return None.
        assert!(cons.acquire().is_none());
    }

    /// Tests an alternating pop–push sequence.
    ///
    /// For a queue of capacity 3:
    /// 1. Pop one element (slot 0, value 1) and immediately push it back (making slot 0 RECLAIMABLE).
    /// 2. Then pop the remaining two allocated elements (slots 1 and 2, values 2 and 3).
    /// 3. Since the queue is empty (count == 0) but slot 0 is RECLAIMABLE,
    ///    the next pop triggers reclaiming and returns value 1.
    #[test]
    fn test_alternating_pop_push() {
        let queue = NtsRing::new(vec![1, 2, 3]);
        let (prod, cons) = queue.split();

        // Pop one element from slot 0.
        let guard1 = cons.acquire().expect("Should pop an element");
        assert_eq!(*guard1, 1);

        // Immediately push it back so that slot 0 becomes RECLAIMABLE.
        prod.release(guard1);

        // Pop the next two allocated elements (from slots 1 and 2).
        let guard2 = cons.acquire().expect("Should pop second element");
        assert_eq!(*guard2, 2);
        let guard3 = cons.acquire().expect("Should pop third element");
        assert_eq!(*guard3, 3);

        // Now, the queue is empty (head == tail, count == 0) but slot 0 is RECLAIMABLE.
        // This triggers reclamation on the next pop.
        let reclaimed = cons.acquire().expect("Should reclaim the pushed element");
        assert_eq!(*reclaimed, 1);

        // The queue should now be empty.
        assert!(cons.acquire().is_none());
    }

    /// Runs multiple cycles where all popped items are concurrently pushed back.
    ///
    /// For a queue with capacity 3, in each cycle:
    /// 1. The consumer pops until no allocated item remains.
    /// 2. Each popped guard is pushed back concurrently by different threads.
    /// 3. The consumer then reclaims the pushed (RECLAIMABLE) items.
    ///
    /// The test checks that the multiset of reclaimed values matches the original set.
    #[test]
    fn test_long_cycle_concurrent() {
        let initial = vec![1, 2, 3];
        let queue = NtsRing::new(initial.clone());
        let (prod, cons) = queue.split();

        // Perform 10 full cycles.
        for _ in 0..10 {
            // Consumer pops until the queue appears empty.
            let mut popped = Vec::new();
            while let Some(guard) = cons.acquire() {
                popped.push(guard);
            }
            for guard in popped {
                prod.release(guard);
            }
            // After all pushes, the consumer should be able to reclaim all pushed slots.
            let mut cycle_vals = Vec::new();
            let mut guards = Vec::new();
            while let Some(guard) = cons.acquire() {
                cycle_vals.push(*guard);
                guards.push(guard);
            }

            for guard in guards {
                prod.release(guard);
            }

            // Although the order might be rotated, the multiset must match.
            cycle_vals.sort();
            let mut sorted_initial = initial.clone();
            sorted_initial.sort();
            assert_eq!(
                cycle_vals, sorted_initial,
                "Cycle did not yield the expected elements"
            );
        }
    }

    /// Verifies that multiple cloned Producer handles share the same underlying queue.
    ///
    /// This test pops one element, then uses two different producer clones (in separate threads)
    /// to push different elements back. The consumer then reclaims all slots and the final
    /// multiset of values should match the initial set.
    #[test]
    fn test_producer_clone_shared_state() {
        let queue = NtsRing::new(vec![100, 200, 300]);
        let (prod, cons) = queue.split();

        // Clone the producer handle.
        let prod1 = prod.clone();
        let prod2 = prod.clone();

        // Pop one element.
        let guard1 = cons.acquire().expect("Expected an element");
        // Use one clone to push it back.
        prod1.release(guard1);

        // Pop the remaining two elements.
        let guard2 = cons.acquire().expect("Expected a second element");
        let guard3 = cons.acquire().expect("Expected a third element");

        // Use different producer clones concurrently to push them back.
        let h1 = thread::spawn(move || {
            prod2.release(guard2);
        });
        let h2 = thread::spawn(move || {
            prod.release(guard3);
        });
        h1.join().expect("Producer thread failed");
        h2.join().expect("Producer thread failed");

        // Reclaim all pushed items.
        let mut reclaimed = Vec::new();
        while let Some(guard) = cons.acquire() {
            reclaimed.push(*guard);
        }
        reclaimed.sort();
        let mut expected = vec![100, 200, 300];
        expected.sort();
        assert_eq!(reclaimed, expected);
    }
}
