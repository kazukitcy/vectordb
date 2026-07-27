// Intrinsic modules are designed for wildcard import; itemizing the NEON
// intrinsic names adds churn without clarifying provenance.
#[allow(clippy::wildcard_imports)]
use std::arch::{aarch64::*, asm};
use std::mem::size_of_val;

use vectordb_core::F16;

// Float kernels process paired four-lane chunks with independent accumulators, combine them once,
// reduce the four lanes once, and then use the scalar formulas for the remainder.
//
// Stable 1.93.1 does not expose the NEON fp16 conversion intrinsics, so the
// `stdarch_neon_f16` work remains deferred. F16 kernels convert through fixed 64-element stack
// chunks before using the same NEON f32 accumulation.
//
// The i8 dot kernel uses vld1_s8 -> vmull_s8 -> vpadalq_s16. Squared L2 uses vld1_s8 ->
// vsubl_s8, then squares the low and high i16x4 halves with vmull_s16 and accumulates them with
// vaddq_s32. At MAX_I8_DIMENSION, each squared-L2 lane accumulator is bounded by
// 4_096 * 65_025 = 266_342_400 (532_684_800 after combining low and high), and each dot
// accumulator lane is bounded by 4_096 * 16_384 = 67_108_864 (134_217_728 after combining the
// paired accumulators). The final reductions are bounded by 32_768 * 65_025 = 2_130_739_200 for
// squared L2 and 32_768 * 16_384 = 536_870_912 for dot, so every i32 operation remains in range.
//
// Full-vector prefetch walks the target in 64-byte cache-line-sized steps; batch scoring bounds
// its automatic hint to the first four lines. Both request temporal L1 retention. ScoreKernel
// dispatches them only for the Neon path; scalar-path prefetch remains a no-op.
const CACHE_LINE_BYTES: usize = 64;
const PREFETCH_START_BYTES: usize = 4 * CACHE_LINE_BYTES;
const F16_CHUNK_ELEMENTS: usize = 64;

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

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports NEON.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn squared_l2_f16(a: &[F16], b: &[F16]) -> f32 {
    let mut left_chunk = [0.0f32; F16_CHUNK_ELEMENTS];
    let mut right_chunk = [0.0f32; F16_CHUNK_ELEMENTS];
    let mut sum = 0.0;
    let mut offset = 0;

    while offset < a.len() {
        let chunk_len = (a.len() - offset).min(F16_CHUNK_ELEMENTS);
        for index in 0..chunk_len {
            left_chunk[index] = a[offset + index].to_f32();
            right_chunk[index] = b[offset + index].to_f32();
        }
        sum += unsafe {
            // SAFETY: both slices have chunk_len elements and this function requires NEON.
            accumulate_squared_l2_f32(&left_chunk[..chunk_len], &right_chunk[..chunk_len])
        };
        offset += chunk_len;
    }
    sum
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports NEON.
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neg_dot_f16(a: &[F16], b: &[F16]) -> f32 {
    let mut left_chunk = [0.0f32; F16_CHUNK_ELEMENTS];
    let mut right_chunk = [0.0f32; F16_CHUNK_ELEMENTS];
    let mut sum = 0.0;
    let mut offset = 0;

    while offset < a.len() {
        let chunk_len = (a.len() - offset).min(F16_CHUNK_ELEMENTS);
        for index in 0..chunk_len {
            left_chunk[index] = a[offset + index].to_f32();
            right_chunk[index] = b[offset + index].to_f32();
        }
        sum += unsafe {
            // SAFETY: both slices have chunk_len elements and this function requires NEON.
            accumulate_dot_f32(&left_chunk[..chunk_len], &right_chunk[..chunk_len])
        };
        offset += chunk_len;
    }
    -sum
}

// Safety preconditions: the caller guarantees equal lengths, a length no greater than
// MAX_I8_DIMENSION, and that the CPU supports NEON.
#[allow(clippy::cast_precision_loss)]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn squared_l2_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut low_accumulator = vdupq_n_s32(0);
    let mut high_accumulator = vdupq_n_s32(0);
    let mut index = 0;

    while a.len() - index >= 8 {
        let left = unsafe {
            // SAFETY: the loop bound provides eight readable bytes; vld1_s8 permits an unaligned
            // address.
            vld1_s8(a.as_ptr().wrapping_add(index))
        };
        let right = unsafe {
            // SAFETY: equal lengths and the loop bound provide eight readable bytes; vld1_s8
            // permits an unaligned address.
            vld1_s8(b.as_ptr().wrapping_add(index))
        };
        let difference = vsubl_s8(left, right);
        let difference_low = vget_low_s16(difference);
        let difference_high = vget_high_s16(difference);
        low_accumulator = vaddq_s32(low_accumulator, vmull_s16(difference_low, difference_low));
        high_accumulator = vaddq_s32(
            high_accumulator,
            vmull_s16(difference_high, difference_high),
        );
        index += 8;
    }

    let mut sum = vaddvq_s32(vaddq_s32(low_accumulator, high_accumulator));
    while index < a.len() {
        let difference = i32::from(a[index]) - i32::from(b[index]);
        sum += difference * difference;
        index += 1;
    }
    sum as f32
}

// Safety preconditions: the caller guarantees equal lengths, a length no greater than
// MAX_I8_DIMENSION, and that the CPU supports NEON.
#[allow(clippy::cast_precision_loss)]
#[target_feature(enable = "neon")]
pub(crate) unsafe fn neg_dot_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut first_accumulator = vdupq_n_s32(0);
    let mut second_accumulator = vdupq_n_s32(0);
    let mut index = 0;

    while a.len() - index >= 16 {
        let left_first = unsafe {
            // SAFETY: the loop bound provides eight readable bytes; vld1_s8 permits an unaligned
            // address.
            vld1_s8(a.as_ptr().wrapping_add(index))
        };
        let right_first = unsafe {
            // SAFETY: equal lengths and the loop bound provide eight readable bytes; vld1_s8
            // permits an unaligned address.
            vld1_s8(b.as_ptr().wrapping_add(index))
        };
        let left_second = unsafe {
            // SAFETY: the loop bound provides eight readable bytes at the second block; vld1_s8
            // permits an unaligned address.
            vld1_s8(a.as_ptr().wrapping_add(index + 8))
        };
        let right_second = unsafe {
            // SAFETY: equal lengths and the loop bound provide eight readable bytes at the second
            // block; vld1_s8 permits an unaligned address.
            vld1_s8(b.as_ptr().wrapping_add(index + 8))
        };
        first_accumulator = vpadalq_s16(first_accumulator, vmull_s8(left_first, right_first));
        second_accumulator = vpadalq_s16(second_accumulator, vmull_s8(left_second, right_second));
        index += 16;
    }

    if a.len() - index >= 8 {
        let left = unsafe {
            // SAFETY: the remainder bound provides eight readable bytes; vld1_s8 permits an
            // unaligned address.
            vld1_s8(a.as_ptr().wrapping_add(index))
        };
        let right = unsafe {
            // SAFETY: equal lengths and the remainder bound provide eight readable bytes; vld1_s8
            // permits an unaligned address.
            vld1_s8(b.as_ptr().wrapping_add(index))
        };
        first_accumulator = vpadalq_s16(first_accumulator, vmull_s8(left, right));
        index += 8;
    }

    let mut sum = vaddvq_s32(vaddq_s32(first_accumulator, second_accumulator));
    while index < a.len() {
        sum += i32::from(a[index]) * i32::from(b[index]);
        index += 1;
    }
    -(sum as f32)
}
