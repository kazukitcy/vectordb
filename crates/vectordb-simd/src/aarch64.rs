// Intrinsic modules are designed for wildcard import; itemizing the NEON
// intrinsic names adds churn without clarifying provenance. allow, not
// expect: clippy suppresses wildcard_imports in test builds, so the
// expectation would be unfulfilled under --all-targets.
#[allow(clippy::wildcard_imports)]
use std::arch::{aarch64::*, asm};
use std::mem::size_of_val;

// This module provides f32 kernels only. Hand-written NEON f16 and i8 kernels
// were removed after the two-reference benchmark record showed them at or
// below the scalar path, which the compiler auto-vectorizes on aarch64 (NEON
// is the baseline ISA; the i8 naive loop measured 2-4x faster than the removed
// widening kernel — see ADR 0002 and the adjudication log). Reintroduction
// requires a measured margin over both the naive and safe baselines.
//
// Float kernels process paired four-lane chunks with independent accumulators, combine them once,
// reduce the four lanes once, and then use the scalar formulas for the remainder.
//
// Full-vector prefetch walks the target in 64-byte cache-line-sized steps; batch scoring bounds
// its automatic hint to the first four lines. Both request temporal L1 retention. ScoreKernel
// dispatches them only for the Neon path; scalar-path prefetch remains a no-op.
const CACHE_LINE_BYTES: usize = 64;
const PREFETCH_START_BYTES: usize = 4 * CACHE_LINE_BYTES;

pub(crate) fn prefetch<T>(target: &[T]) {
    prefetch_bytes(target, size_of_val(target));
}

pub(crate) fn prefetch_start<T>(target: &[T]) {
    prefetch_bytes(target, size_of_val(target).min(PREFETCH_START_BYTES));
}

fn prefetch_bytes<T>(target: &[T], byte_len: usize) {
    let start = target.as_ptr().cast::<u8>();
    for offset in (0..byte_len).step_by(CACHE_LINE_BYTES) {
        let address = start.wrapping_add(offset);
        unsafe {
            // SAFETY: every offset is strictly below the slice's byte length, so the address is
            // backed by the target slice. PRFM is a read-only architectural hint.
            asm!(
                "prfm pldl1keep, [{0}]",
                in(reg) address,
                options(nostack, preserves_flags, readonly)
            );
        }
    }
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports NEON.
#[target_feature(enable = "neon")]
unsafe fn accumulate_squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut first_accumulator = vdupq_n_f32(0.0);
    let mut second_accumulator = vdupq_n_f32(0.0);
    let mut index = 0;

    while a.len() - index >= 8 {
        let left_first = unsafe {
            // SAFETY: the loop bound leaves four f32 elements at this offset.
            vld1q_f32(a.as_ptr().wrapping_add(index))
        };
        let right_first = unsafe {
            // SAFETY: equal lengths and the loop bound leave four f32 elements at this offset.
            vld1q_f32(b.as_ptr().wrapping_add(index))
        };
        let left_second = unsafe {
            // SAFETY: the loop bound leaves four f32 elements at this offset.
            vld1q_f32(a.as_ptr().wrapping_add(index + 4))
        };
        let right_second = unsafe {
            // SAFETY: equal lengths and the loop bound leave four f32 elements at this offset.
            vld1q_f32(b.as_ptr().wrapping_add(index + 4))
        };

        let first_difference = vsubq_f32(left_first, right_first);
        let second_difference = vsubq_f32(left_second, right_second);
        first_accumulator = vfmaq_f32(first_accumulator, first_difference, first_difference);
        second_accumulator = vfmaq_f32(second_accumulator, second_difference, second_difference);
        index += 8;
    }

    if a.len() - index >= 4 {
        let left = unsafe {
            // SAFETY: the remainder bound leaves four f32 elements at this offset.
            vld1q_f32(a.as_ptr().wrapping_add(index))
        };
        let right = unsafe {
            // SAFETY: equal lengths and the remainder bound leave four f32 elements here.
            vld1q_f32(b.as_ptr().wrapping_add(index))
        };
        let difference = vsubq_f32(left, right);
        first_accumulator = vfmaq_f32(first_accumulator, difference, difference);
        index += 4;
    }

    let mut sum = vaddvq_f32(vaddq_f32(first_accumulator, second_accumulator));
    while index < a.len() {
        let difference = a[index] - b[index];
        sum += difference * difference;
        index += 1;
    }
    sum
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports NEON.
#[target_feature(enable = "neon")]
unsafe fn accumulate_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut first_accumulator = vdupq_n_f32(0.0);
    let mut second_accumulator = vdupq_n_f32(0.0);
    let mut index = 0;

    while a.len() - index >= 8 {
        let left_first = unsafe {
            // SAFETY: the loop bound leaves four f32 elements at this offset.
            vld1q_f32(a.as_ptr().wrapping_add(index))
        };
        let right_first = unsafe {
            // SAFETY: equal lengths and the loop bound leave four f32 elements at this offset.
            vld1q_f32(b.as_ptr().wrapping_add(index))
        };
        let left_second = unsafe {
            // SAFETY: the loop bound leaves four f32 elements at this offset.
            vld1q_f32(a.as_ptr().wrapping_add(index + 4))
        };
        let right_second = unsafe {
            // SAFETY: equal lengths and the loop bound leave four f32 elements at this offset.
            vld1q_f32(b.as_ptr().wrapping_add(index + 4))
        };

        first_accumulator = vfmaq_f32(first_accumulator, left_first, right_first);
        second_accumulator = vfmaq_f32(second_accumulator, left_second, right_second);
        index += 8;
    }

    if a.len() - index >= 4 {
        let left = unsafe {
            // SAFETY: the remainder bound leaves four f32 elements at this offset.
            vld1q_f32(a.as_ptr().wrapping_add(index))
        };
        let right = unsafe {
            // SAFETY: equal lengths and the remainder bound leave four f32 elements here.
            vld1q_f32(b.as_ptr().wrapping_add(index))
        };
        first_accumulator = vfmaq_f32(first_accumulator, left, right);
        index += 4;
    }

    let mut sum = vaddvq_f32(vaddq_f32(first_accumulator, second_accumulator));
    while index < a.len() {
        sum += a[index] * b[index];
        index += 1;
    }
    sum
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports NEON.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    unsafe {
        // SAFETY: the caller provides the helper's equal-length and NEON preconditions.
        accumulate_squared_l2_f32(a, b)
    }
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports NEON.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neg_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot = unsafe {
        // SAFETY: the caller provides the helper's equal-length and NEON preconditions.
        accumulate_dot_f32(a, b)
    };
    -dot
}
