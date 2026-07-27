// Intrinsic modules are designed for wildcard import; itemizing dozens of
// intrinsic names adds churn without clarifying provenance.
#[allow(clippy::wildcard_imports)]
use std::arch::x86_64::*;
use std::mem::size_of_val;

use vectordb_core::F16;

// AVX2 float kernels process paired eight-lane chunks with independent accumulators, combine them
// once, reduce the eight lanes once, and then use the scalar formulas for the remainder. AVX2
// integer kernels do the same with paired sixteen-byte chunks widened into i32 lane accumulators,
// keeping the scalar remainder in i32 until the single final f32 conversion.
//
// Full-vector prefetch walks the target in 64-byte cache-line-sized steps; batch scoring bounds
// its automatic hint to the first four lines. Both request temporal L1 retention. ScoreKernel
// dispatches them only for x86 SIMD paths; scalar-path prefetch remains a no-op.
const CACHE_LINE_BYTES: usize = 64;
const PREFETCH_START_BYTES: usize = 4 * CACHE_LINE_BYTES;

pub(crate) fn prefetch<T>(target: &[T]) {
    prefetch_bytes(target, size_of_val(target));
}

pub(crate) fn prefetch_start<T>(target: &[T]) {
    prefetch_bytes(target, size_of_val(target).min(PREFETCH_START_BYTES));
}

fn prefetch_bytes<T>(target: &[T], byte_len: usize) {
    let start = target.as_ptr().cast::<i8>();
    for offset in (0..byte_len).step_by(CACHE_LINE_BYTES) {
        unsafe {
            // SAFETY: SSE is part of the x86_64 baseline, and prefetch is an
            // architectural hint that is valid for any address.
            _mm_prefetch::<_MM_HINT_T0>(start.wrapping_add(offset));
        }
    }
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports AVX2 and
// FMA.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut first_accumulator = _mm256_setzero_ps();
    let mut second_accumulator = _mm256_setzero_ps();
    let mut index = 0;

    while a.len() - index >= 16 {
        let left_first = unsafe {
            // SAFETY: the loop bound leaves eight f32 elements at this offset.
            _mm256_loadu_ps(a.as_ptr().wrapping_add(index))
        };
        let right_first = unsafe {
            // SAFETY: equal lengths and the loop bound leave eight f32 elements at this offset.
            _mm256_loadu_ps(b.as_ptr().wrapping_add(index))
        };
        let left_second = unsafe {
            // SAFETY: the loop bound leaves eight f32 elements at this offset.
            _mm256_loadu_ps(a.as_ptr().wrapping_add(index + 8))
        };
        let right_second = unsafe {
            // SAFETY: equal lengths and the loop bound leave eight f32 elements at this offset.
            _mm256_loadu_ps(b.as_ptr().wrapping_add(index + 8))
        };

        let first_difference = _mm256_sub_ps(left_first, right_first);
        let second_difference = _mm256_sub_ps(left_second, right_second);
        first_accumulator = _mm256_fmadd_ps(first_difference, first_difference, first_accumulator);
        second_accumulator =
            _mm256_fmadd_ps(second_difference, second_difference, second_accumulator);
        index += 16;
    }

    if a.len() - index >= 8 {
        let left = unsafe {
            // SAFETY: the remainder bound leaves eight f32 elements at this offset.
            _mm256_loadu_ps(a.as_ptr().wrapping_add(index))
        };
        let right = unsafe {
            // SAFETY: equal lengths and the remainder bound leave eight f32 elements here.
            _mm256_loadu_ps(b.as_ptr().wrapping_add(index))
        };
        let difference = _mm256_sub_ps(left, right);
        first_accumulator = _mm256_fmadd_ps(difference, difference, first_accumulator);
        index += 8;
    }

    let vector_sum = _mm256_add_ps(first_accumulator, second_accumulator);
    let low = _mm256_castps256_ps128(vector_sum);
    let high = _mm256_extractf128_ps::<1>(vector_sum);
    let pairs = _mm_add_ps(low, high);
    let pair_high = _mm_movehl_ps(pairs, pairs);
    let quads = _mm_add_ps(pairs, pair_high);
    let quad_high = _mm_shuffle_ps::<0x55>(quads, quads);
    let mut sum = _mm_cvtss_f32(_mm_add_ss(quads, quad_high));

    while index < a.len() {
        let difference = a[index] - b[index];
        sum += difference * difference;
        index += 1;
    }
    sum
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports AVX2 and
// FMA.
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn neg_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut first_accumulator = _mm256_setzero_ps();
    let mut second_accumulator = _mm256_setzero_ps();
    let mut index = 0;

    while a.len() - index >= 16 {
        let left_first = unsafe {
            // SAFETY: the loop bound leaves eight f32 elements at this offset.
            _mm256_loadu_ps(a.as_ptr().wrapping_add(index))
        };
        let right_first = unsafe {
            // SAFETY: equal lengths and the loop bound leave eight f32 elements at this offset.
            _mm256_loadu_ps(b.as_ptr().wrapping_add(index))
        };
        let left_second = unsafe {
            // SAFETY: the loop bound leaves eight f32 elements at this offset.
            _mm256_loadu_ps(a.as_ptr().wrapping_add(index + 8))
        };
        let right_second = unsafe {
            // SAFETY: equal lengths and the loop bound leave eight f32 elements at this offset.
            _mm256_loadu_ps(b.as_ptr().wrapping_add(index + 8))
        };

        first_accumulator = _mm256_fmadd_ps(left_first, right_first, first_accumulator);
        second_accumulator = _mm256_fmadd_ps(left_second, right_second, second_accumulator);
        index += 16;
    }

    if a.len() - index >= 8 {
        let left = unsafe {
            // SAFETY: the remainder bound leaves eight f32 elements at this offset.
            _mm256_loadu_ps(a.as_ptr().wrapping_add(index))
        };
        let right = unsafe {
            // SAFETY: equal lengths and the remainder bound leave eight f32 elements here.
            _mm256_loadu_ps(b.as_ptr().wrapping_add(index))
        };
        first_accumulator = _mm256_fmadd_ps(left, right, first_accumulator);
        index += 8;
    }

    let vector_sum = _mm256_add_ps(first_accumulator, second_accumulator);
    let low = _mm256_castps256_ps128(vector_sum);
    let high = _mm256_extractf128_ps::<1>(vector_sum);
    let pairs = _mm_add_ps(low, high);
    let pair_high = _mm_movehl_ps(pairs, pairs);
    let quads = _mm_add_ps(pairs, pair_high);
    let quad_high = _mm_shuffle_ps::<0x55>(quads, quads);
    let mut sum = _mm_cvtss_f32(_mm_add_ss(quads, quad_high));

    while index < a.len() {
        sum += a[index] * b[index];
        index += 1;
    }
    -sum
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports AVX2, FMA,
// and F16C.
#[allow(clippy::cast_ptr_alignment)]
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn squared_l2_f16(a: &[F16], b: &[F16]) -> f32 {
    let mut first_accumulator = _mm256_setzero_ps();
    let mut second_accumulator = _mm256_setzero_ps();
    let mut index = 0;

    while a.len() - index >= 16 {
        let left_first_bits = unsafe {
            // SAFETY: F16's layout contract and the loop bound provide 16 readable bytes; the
            // intrinsic permits an unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let right_first_bits = unsafe {
            // SAFETY: equal lengths, F16's layout contract, and the loop bound provide 16 readable
            // bytes; the intrinsic permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let left_second_bits = unsafe {
            // SAFETY: F16's layout contract and the loop bound provide 16 readable bytes; the
            // intrinsic permits an unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index + 8).cast::<__m128i>())
        };
        let right_second_bits = unsafe {
            // SAFETY: equal lengths, F16's layout contract, and the loop bound provide 16 readable
            // bytes; the intrinsic permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index + 8).cast::<__m128i>())
        };

        let left_first = _mm256_cvtph_ps(left_first_bits);
        let right_first = _mm256_cvtph_ps(right_first_bits);
        let left_second = _mm256_cvtph_ps(left_second_bits);
        let right_second = _mm256_cvtph_ps(right_second_bits);
        let first_difference = _mm256_sub_ps(left_first, right_first);
        let second_difference = _mm256_sub_ps(left_second, right_second);
        first_accumulator = _mm256_fmadd_ps(first_difference, first_difference, first_accumulator);
        second_accumulator =
            _mm256_fmadd_ps(second_difference, second_difference, second_accumulator);
        index += 16;
    }

    if a.len() - index >= 8 {
        let left_bits = unsafe {
            // SAFETY: F16's layout contract and the remainder bound provide 16 readable bytes; the
            // intrinsic permits an unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let right_bits = unsafe {
            // SAFETY: equal lengths, F16's layout contract, and the remainder bound provide 16
            // readable bytes; the intrinsic permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let left = _mm256_cvtph_ps(left_bits);
        let right = _mm256_cvtph_ps(right_bits);
        let difference = _mm256_sub_ps(left, right);
        first_accumulator = _mm256_fmadd_ps(difference, difference, first_accumulator);
        index += 8;
    }

    let vector_sum = _mm256_add_ps(first_accumulator, second_accumulator);
    let low = _mm256_castps256_ps128(vector_sum);
    let high = _mm256_extractf128_ps::<1>(vector_sum);
    let pairs = _mm_add_ps(low, high);
    let pair_high = _mm_movehl_ps(pairs, pairs);
    let quads = _mm_add_ps(pairs, pair_high);
    let quad_high = _mm_shuffle_ps::<0x55>(quads, quads);
    let mut sum = _mm_cvtss_f32(_mm_add_ss(quads, quad_high));

    while index < a.len() {
        let difference = a[index].to_f32() - b[index].to_f32();
        sum += difference * difference;
        index += 1;
    }
    sum
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports AVX2, FMA,
// and F16C.
#[allow(clippy::cast_ptr_alignment)]
#[target_feature(enable = "avx2,fma,f16c")]
pub(crate) unsafe fn neg_dot_f16(a: &[F16], b: &[F16]) -> f32 {
    let mut first_accumulator = _mm256_setzero_ps();
    let mut second_accumulator = _mm256_setzero_ps();
    let mut index = 0;

    while a.len() - index >= 16 {
        let left_first_bits = unsafe {
            // SAFETY: F16's layout contract and the loop bound provide 16 readable bytes; the
            // intrinsic permits an unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let right_first_bits = unsafe {
            // SAFETY: equal lengths, F16's layout contract, and the loop bound provide 16 readable
            // bytes; the intrinsic permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let left_second_bits = unsafe {
            // SAFETY: F16's layout contract and the loop bound provide 16 readable bytes; the
            // intrinsic permits an unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index + 8).cast::<__m128i>())
        };
        let right_second_bits = unsafe {
            // SAFETY: equal lengths, F16's layout contract, and the loop bound provide 16 readable
            // bytes; the intrinsic permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index + 8).cast::<__m128i>())
        };

        let left_first = _mm256_cvtph_ps(left_first_bits);
        let right_first = _mm256_cvtph_ps(right_first_bits);
        let left_second = _mm256_cvtph_ps(left_second_bits);
        let right_second = _mm256_cvtph_ps(right_second_bits);
        first_accumulator = _mm256_fmadd_ps(left_first, right_first, first_accumulator);
        second_accumulator = _mm256_fmadd_ps(left_second, right_second, second_accumulator);
        index += 16;
    }

    if a.len() - index >= 8 {
        let left_bits = unsafe {
            // SAFETY: F16's layout contract and the remainder bound provide 16 readable bytes; the
            // intrinsic permits an unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let right_bits = unsafe {
            // SAFETY: equal lengths, F16's layout contract, and the remainder bound provide 16
            // readable bytes; the intrinsic permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let left = _mm256_cvtph_ps(left_bits);
        let right = _mm256_cvtph_ps(right_bits);
        first_accumulator = _mm256_fmadd_ps(left, right, first_accumulator);
        index += 8;
    }

    let vector_sum = _mm256_add_ps(first_accumulator, second_accumulator);
    let low = _mm256_castps256_ps128(vector_sum);
    let high = _mm256_extractf128_ps::<1>(vector_sum);
    let pairs = _mm_add_ps(low, high);
    let pair_high = _mm_movehl_ps(pairs, pairs);
    let quads = _mm_add_ps(pairs, pair_high);
    let quad_high = _mm_shuffle_ps::<0x55>(quads, quads);
    let mut sum = _mm_cvtss_f32(_mm_add_ss(quads, quad_high));

    while index < a.len() {
        sum += a[index].to_f32() * b[index].to_f32();
        index += 1;
    }
    -sum
}

// Safety preconditions: the caller guarantees equal lengths, a length no greater than
// MAX_I8_DIMENSION, and that the CPU supports AVX2.
#[allow(clippy::cast_ptr_alignment, clippy::cast_precision_loss)]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn squared_l2_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut first_accumulator = _mm256_setzero_si256();
    let mut second_accumulator = _mm256_setzero_si256();
    let mut index = 0;

    while a.len() - index >= 32 {
        let left_first_bytes = unsafe {
            // SAFETY: the loop bound provides 16 readable bytes; the intrinsic permits an
            // unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let right_first_bytes = unsafe {
            // SAFETY: equal lengths and the loop bound provide 16 readable bytes; the intrinsic
            // permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let left_second_bytes = unsafe {
            // SAFETY: the loop bound provides 16 readable bytes; the intrinsic permits an
            // unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index + 16).cast::<__m128i>())
        };
        let right_second_bytes = unsafe {
            // SAFETY: equal lengths and the loop bound provide 16 readable bytes; the intrinsic
            // permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index + 16).cast::<__m128i>())
        };

        let left_first = _mm256_cvtepi8_epi16(left_first_bytes);
        let right_first = _mm256_cvtepi8_epi16(right_first_bytes);
        let left_second = _mm256_cvtepi8_epi16(left_second_bytes);
        let right_second = _mm256_cvtepi8_epi16(right_second_bytes);
        let first_difference = _mm256_sub_epi16(left_first, right_first);
        let second_difference = _mm256_sub_epi16(left_second, right_second);
        first_accumulator = _mm256_add_epi32(
            first_accumulator,
            _mm256_madd_epi16(first_difference, first_difference),
        );
        second_accumulator = _mm256_add_epi32(
            second_accumulator,
            _mm256_madd_epi16(second_difference, second_difference),
        );
        index += 32;
    }

    if a.len() - index >= 16 {
        let left_bytes = unsafe {
            // SAFETY: the remainder bound provides 16 readable bytes; the intrinsic permits an
            // unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let right_bytes = unsafe {
            // SAFETY: equal lengths and the remainder bound provide 16 readable bytes; the
            // intrinsic permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let left = _mm256_cvtepi8_epi16(left_bytes);
        let right = _mm256_cvtepi8_epi16(right_bytes);
        let difference = _mm256_sub_epi16(left, right);
        first_accumulator =
            _mm256_add_epi32(first_accumulator, _mm256_madd_epi16(difference, difference));
        index += 16;
    }

    let vector_sum = _mm256_add_epi32(first_accumulator, second_accumulator);
    let low = _mm256_castsi256_si128(vector_sum);
    let high = _mm256_extracti128_si256::<1>(vector_sum);
    let pairs = _mm_add_epi32(low, high);
    let quads = _mm_add_epi32(pairs, _mm_shuffle_epi32::<0x4e>(pairs));
    let mut sum = _mm_cvtsi128_si32(_mm_add_epi32(quads, _mm_shuffle_epi32::<0xb1>(quads)));

    while index < a.len() {
        let difference = i32::from(a[index]) - i32::from(b[index]);
        sum += difference * difference;
        index += 1;
    }
    sum as f32
}

// Safety preconditions: the caller guarantees equal lengths, a length no greater than
// MAX_I8_DIMENSION, and that the CPU supports AVX2.
#[allow(clippy::cast_ptr_alignment, clippy::cast_precision_loss)]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn neg_dot_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut first_accumulator = _mm256_setzero_si256();
    let mut second_accumulator = _mm256_setzero_si256();
    let mut index = 0;

    while a.len() - index >= 32 {
        let left_first_bytes = unsafe {
            // SAFETY: the loop bound provides 16 readable bytes; the intrinsic permits an
            // unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let right_first_bytes = unsafe {
            // SAFETY: equal lengths and the loop bound provide 16 readable bytes; the intrinsic
            // permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let left_second_bytes = unsafe {
            // SAFETY: the loop bound provides 16 readable bytes; the intrinsic permits an
            // unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index + 16).cast::<__m128i>())
        };
        let right_second_bytes = unsafe {
            // SAFETY: equal lengths and the loop bound provide 16 readable bytes; the intrinsic
            // permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index + 16).cast::<__m128i>())
        };

        let left_first = _mm256_cvtepi8_epi16(left_first_bytes);
        let right_first = _mm256_cvtepi8_epi16(right_first_bytes);
        let left_second = _mm256_cvtepi8_epi16(left_second_bytes);
        let right_second = _mm256_cvtepi8_epi16(right_second_bytes);
        first_accumulator = _mm256_add_epi32(
            first_accumulator,
            _mm256_madd_epi16(left_first, right_first),
        );
        second_accumulator = _mm256_add_epi32(
            second_accumulator,
            _mm256_madd_epi16(left_second, right_second),
        );
        index += 32;
    }

    if a.len() - index >= 16 {
        let left_bytes = unsafe {
            // SAFETY: the remainder bound provides 16 readable bytes; the intrinsic permits an
            // unaligned address.
            _mm_loadu_si128(a.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let right_bytes = unsafe {
            // SAFETY: equal lengths and the remainder bound provide 16 readable bytes; the
            // intrinsic permits an unaligned address.
            _mm_loadu_si128(b.as_ptr().wrapping_add(index).cast::<__m128i>())
        };
        let left = _mm256_cvtepi8_epi16(left_bytes);
        let right = _mm256_cvtepi8_epi16(right_bytes);
        first_accumulator = _mm256_add_epi32(first_accumulator, _mm256_madd_epi16(left, right));
        index += 16;
    }

    let vector_sum = _mm256_add_epi32(first_accumulator, second_accumulator);
    let low = _mm256_castsi256_si128(vector_sum);
    let high = _mm256_extracti128_si256::<1>(vector_sum);
    let pairs = _mm_add_epi32(low, high);
    let quads = _mm_add_epi32(pairs, _mm_shuffle_epi32::<0x4e>(pairs));
    let mut sum = _mm_cvtsi128_si32(_mm_add_epi32(quads, _mm_shuffle_epi32::<0xb1>(quads)));

    while index < a.len() {
        sum += i32::from(a[index]) * i32::from(b[index]);
        index += 1;
    }
    -(sum as f32)
}

// Safety preconditions: `index..index + 16` is in bounds for `values`, and the CPU supports
// AVX-512F and F16C.
#[inline]
#[allow(clippy::cast_ptr_alignment)]
#[target_feature(enable = "avx512f,f16c")]
unsafe fn load_f16x16(values: &[F16], index: usize) -> __m512 {
    let low_bits = unsafe {
        // SAFETY: the caller provides 16 readable F16 values from `index`; F16's layout contract
        // makes the first eight values 16 readable bytes, and the intrinsic permits an unaligned
        // address.
        _mm_loadu_si128(values.as_ptr().wrapping_add(index).cast::<__m128i>())
    };
    let high_bits = unsafe {
        // SAFETY: the caller provides 16 readable F16 values from `index`; F16's layout contract
        // makes the second eight values 16 readable bytes, and the intrinsic permits an unaligned
        // address.
        _mm_loadu_si128(values.as_ptr().wrapping_add(index + 8).cast::<__m128i>())
    };
    let low = _mm256_cvtph_ps(low_bits);
    let high = _mm256_cvtph_ps(high_bits);
    _mm512_shuffle_f32x4::<0x44>(_mm512_castps256_ps512(low), _mm512_castps256_ps512(high))
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports AVX-512F.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn squared_l2_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    let mut first_accumulator = _mm512_setzero_ps();
    let mut second_accumulator = _mm512_setzero_ps();
    let mut index = 0;

    while a.len() - index >= 32 {
        let left_first = unsafe {
            // SAFETY: the loop bound leaves 16 f32 elements at this offset.
            _mm512_loadu_ps(a.as_ptr().wrapping_add(index))
        };
        let right_first = unsafe {
            // SAFETY: equal lengths and the loop bound leave 16 f32 elements at this offset.
            _mm512_loadu_ps(b.as_ptr().wrapping_add(index))
        };
        let left_second = unsafe {
            // SAFETY: the loop bound leaves 16 f32 elements at this offset.
            _mm512_loadu_ps(a.as_ptr().wrapping_add(index + 16))
        };
        let right_second = unsafe {
            // SAFETY: equal lengths and the loop bound leave 16 f32 elements at this offset.
            _mm512_loadu_ps(b.as_ptr().wrapping_add(index + 16))
        };

        let first_difference = _mm512_sub_ps(left_first, right_first);
        let second_difference = _mm512_sub_ps(left_second, right_second);
        first_accumulator = _mm512_fmadd_ps(first_difference, first_difference, first_accumulator);
        second_accumulator =
            _mm512_fmadd_ps(second_difference, second_difference, second_accumulator);
        index += 32;
    }

    if a.len() - index >= 16 {
        let left = unsafe {
            // SAFETY: the remainder bound leaves 16 f32 elements at this offset.
            _mm512_loadu_ps(a.as_ptr().wrapping_add(index))
        };
        let right = unsafe {
            // SAFETY: equal lengths and the remainder bound leave 16 f32 elements here.
            _mm512_loadu_ps(b.as_ptr().wrapping_add(index))
        };
        let difference = _mm512_sub_ps(left, right);
        first_accumulator = _mm512_fmadd_ps(difference, difference, first_accumulator);
        index += 16;
    }

    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(first_accumulator, second_accumulator));
    while index < a.len() {
        let difference = a[index] - b[index];
        sum += difference * difference;
        index += 1;
    }
    sum
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports AVX-512F.
#[target_feature(enable = "avx512f")]
pub(crate) unsafe fn neg_dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    let mut first_accumulator = _mm512_setzero_ps();
    let mut second_accumulator = _mm512_setzero_ps();
    let mut index = 0;

    while a.len() - index >= 32 {
        let left_first = unsafe {
            // SAFETY: the loop bound leaves 16 f32 elements at this offset.
            _mm512_loadu_ps(a.as_ptr().wrapping_add(index))
        };
        let right_first = unsafe {
            // SAFETY: equal lengths and the loop bound leave 16 f32 elements at this offset.
            _mm512_loadu_ps(b.as_ptr().wrapping_add(index))
        };
        let left_second = unsafe {
            // SAFETY: the loop bound leaves 16 f32 elements at this offset.
            _mm512_loadu_ps(a.as_ptr().wrapping_add(index + 16))
        };
        let right_second = unsafe {
            // SAFETY: equal lengths and the loop bound leave 16 f32 elements at this offset.
            _mm512_loadu_ps(b.as_ptr().wrapping_add(index + 16))
        };

        first_accumulator = _mm512_fmadd_ps(left_first, right_first, first_accumulator);
        second_accumulator = _mm512_fmadd_ps(left_second, right_second, second_accumulator);
        index += 32;
    }

    if a.len() - index >= 16 {
        let left = unsafe {
            // SAFETY: the remainder bound leaves 16 f32 elements at this offset.
            _mm512_loadu_ps(a.as_ptr().wrapping_add(index))
        };
        let right = unsafe {
            // SAFETY: equal lengths and the remainder bound leave 16 f32 elements here.
            _mm512_loadu_ps(b.as_ptr().wrapping_add(index))
        };
        first_accumulator = _mm512_fmadd_ps(left, right, first_accumulator);
        index += 16;
    }

    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(first_accumulator, second_accumulator));
    while index < a.len() {
        sum += a[index] * b[index];
        index += 1;
    }
    -sum
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports AVX-512F
// and F16C.
#[target_feature(enable = "avx512f,f16c")]
pub(crate) unsafe fn squared_l2_f16_avx512(a: &[F16], b: &[F16]) -> f32 {
    let mut first_accumulator = _mm512_setzero_ps();
    let mut second_accumulator = _mm512_setzero_ps();
    let mut index = 0;

    while a.len() - index >= 32 {
        let left_first = unsafe {
            // SAFETY: the loop bound leaves 16 elements, and this function requires the helper's
            // complete CPU feature set.
            load_f16x16(a, index)
        };
        let right_first = unsafe {
            // SAFETY: equal lengths and the loop bound leave 16 elements, and this function
            // requires the helper's complete CPU feature set.
            load_f16x16(b, index)
        };
        let left_second = unsafe {
            // SAFETY: the loop bound leaves 16 elements, and this function requires the helper's
            // complete CPU feature set.
            load_f16x16(a, index + 16)
        };
        let right_second = unsafe {
            // SAFETY: equal lengths and the loop bound leave 16 elements, and this function
            // requires the helper's complete CPU feature set.
            load_f16x16(b, index + 16)
        };

        let first_difference = _mm512_sub_ps(left_first, right_first);
        let second_difference = _mm512_sub_ps(left_second, right_second);
        first_accumulator = _mm512_fmadd_ps(first_difference, first_difference, first_accumulator);
        second_accumulator =
            _mm512_fmadd_ps(second_difference, second_difference, second_accumulator);
        index += 32;
    }

    if a.len() - index >= 16 {
        let left = unsafe {
            // SAFETY: the remainder bound leaves 16 elements, and this function requires the
            // helper's complete CPU feature set.
            load_f16x16(a, index)
        };
        let right = unsafe {
            // SAFETY: equal lengths and the remainder bound leave 16 elements, and this function
            // requires the helper's complete CPU feature set.
            load_f16x16(b, index)
        };
        let difference = _mm512_sub_ps(left, right);
        first_accumulator = _mm512_fmadd_ps(difference, difference, first_accumulator);
        index += 16;
    }

    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(first_accumulator, second_accumulator));
    while index < a.len() {
        let difference = a[index].to_f32() - b[index].to_f32();
        sum += difference * difference;
        index += 1;
    }
    sum
}

// Safety preconditions: the caller guarantees equal lengths and that the CPU supports AVX-512F
// and F16C.
#[target_feature(enable = "avx512f,f16c")]
pub(crate) unsafe fn neg_dot_f16_avx512(a: &[F16], b: &[F16]) -> f32 {
    let mut first_accumulator = _mm512_setzero_ps();
    let mut second_accumulator = _mm512_setzero_ps();
    let mut index = 0;

    while a.len() - index >= 32 {
        let left_first = unsafe {
            // SAFETY: the loop bound leaves 16 elements, and this function requires the helper's
            // complete CPU feature set.
            load_f16x16(a, index)
        };
        let right_first = unsafe {
            // SAFETY: equal lengths and the loop bound leave 16 elements, and this function
            // requires the helper's complete CPU feature set.
            load_f16x16(b, index)
        };
        let left_second = unsafe {
            // SAFETY: the loop bound leaves 16 elements, and this function requires the helper's
            // complete CPU feature set.
            load_f16x16(a, index + 16)
        };
        let right_second = unsafe {
            // SAFETY: equal lengths and the loop bound leave 16 elements, and this function
            // requires the helper's complete CPU feature set.
            load_f16x16(b, index + 16)
        };

        first_accumulator = _mm512_fmadd_ps(left_first, right_first, first_accumulator);
        second_accumulator = _mm512_fmadd_ps(left_second, right_second, second_accumulator);
        index += 32;
    }

    if a.len() - index >= 16 {
        let left = unsafe {
            // SAFETY: the remainder bound leaves 16 elements, and this function requires the
            // helper's complete CPU feature set.
            load_f16x16(a, index)
        };
        let right = unsafe {
            // SAFETY: equal lengths and the remainder bound leave 16 elements, and this function
            // requires the helper's complete CPU feature set.
            load_f16x16(b, index)
        };
        first_accumulator = _mm512_fmadd_ps(left, right, first_accumulator);
        index += 16;
    }

    let mut sum = _mm512_reduce_add_ps(_mm512_add_ps(first_accumulator, second_accumulator));
    while index < a.len() {
        sum += a[index].to_f32() * b[index].to_f32();
        index += 1;
    }
    -sum
}

// Safety preconditions: the caller guarantees equal lengths, a length no greater than
// MAX_I8_DIMENSION, and that the CPU supports AVX-512F, AVX-512BW, and AVX-512VNNI.
#[allow(clippy::cast_precision_loss, clippy::cast_ptr_alignment)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(crate) unsafe fn squared_l2_i8_avx512(a: &[i8], b: &[i8]) -> f32 {
    let mut first_accumulator = _mm512_setzero_si512();
    let mut second_accumulator = _mm512_setzero_si512();
    let mut index = 0;

    while a.len() - index >= 64 {
        let left_bytes = unsafe {
            // SAFETY: the loop bound provides 64 readable bytes; the intrinsic permits an
            // unaligned address.
            _mm512_loadu_si512(a.as_ptr().wrapping_add(index).cast::<__m512i>())
        };
        let right_bytes = unsafe {
            // SAFETY: equal lengths and the loop bound provide 64 readable bytes; the intrinsic
            // permits an unaligned address.
            _mm512_loadu_si512(b.as_ptr().wrapping_add(index).cast::<__m512i>())
        };

        let left_first = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(left_bytes));
        let right_first = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(right_bytes));
        let left_second = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(left_bytes));
        let right_second = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64::<1>(right_bytes));
        let first_difference = _mm512_sub_epi16(left_first, right_first);
        let second_difference = _mm512_sub_epi16(left_second, right_second);
        first_accumulator = _mm512_add_epi32(
            first_accumulator,
            _mm512_madd_epi16(first_difference, first_difference),
        );
        second_accumulator = _mm512_add_epi32(
            second_accumulator,
            _mm512_madd_epi16(second_difference, second_difference),
        );
        index += 64;
    }

    let mut sum = _mm512_reduce_add_epi32(_mm512_add_epi32(first_accumulator, second_accumulator));
    while index < a.len() {
        let difference = i32::from(a[index]) - i32::from(b[index]);
        sum += difference * difference;
        index += 1;
    }
    sum as f32
}

// Safety preconditions: the caller guarantees equal lengths, a length no greater than
// MAX_I8_DIMENSION, and that the CPU supports AVX-512F, AVX-512BW, and AVX-512VNNI.
#[allow(clippy::cast_precision_loss, clippy::cast_ptr_alignment)]
#[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
pub(crate) unsafe fn neg_dot_i8_avx512(a: &[i8], b: &[i8]) -> f32 {
    let mut dpbusd_accumulator = _mm512_setzero_si512();
    let mut target_sum_accumulator = _mm512_setzero_si512();
    let query_bias = _mm512_set1_epi8(i8::MIN);
    let unsigned_ones = _mm512_set1_epi8(1);
    let mut index = 0;

    while a.len() - index >= 64 {
        let query_bytes = unsafe {
            // SAFETY: the loop bound provides 64 readable query bytes; the intrinsic permits an
            // unaligned address.
            _mm512_loadu_si512(a.as_ptr().wrapping_add(index).cast::<__m512i>())
        };
        let target_bytes = unsafe {
            // SAFETY: equal lengths and the loop bound provide 64 readable target bytes; the
            // intrinsic permits an unaligned address.
            _mm512_loadu_si512(b.as_ptr().wrapping_add(index).cast::<__m512i>())
        };
        let unsigned_query = _mm512_xor_si512(query_bytes, query_bias);
        dpbusd_accumulator = _mm512_dpbusd_epi32(dpbusd_accumulator, unsigned_query, target_bytes);
        target_sum_accumulator =
            _mm512_dpbusd_epi32(target_sum_accumulator, unsigned_ones, target_bytes);
        index += 64;
    }

    let mut dpbusd_total = _mm512_reduce_add_epi32(dpbusd_accumulator);
    let mut target_sum = _mm512_reduce_add_epi32(target_sum_accumulator);
    while index < a.len() {
        let unsigned_query = u8::from_ne_bytes(a[index].to_ne_bytes()) ^ 0x80;
        let target = i32::from(b[index]);
        dpbusd_total += i32::from(unsigned_query) * target;
        target_sum += target;
        index += 1;
    }

    let dot = dpbusd_total - 128 * target_sum;
    -(dot as f32)
}
