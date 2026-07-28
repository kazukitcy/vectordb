// The naive family measures straightforward, order-preserving loops. The safe
// family measures what safe Rust can reach through auto-vectorized independent
// accumulators. Unsafe kernels are justified only by their margin over safe_*.
// Integer sums are associative within supported dimensions, so naive i8 loops
// may auto-vectorize; that is the honest scalar baseline for i8.

#[inline(never)]
pub fn naive_squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        let difference = a[index] - b[index];
        sum += difference * difference;
    }
    sum
}

#[inline(never)]
pub fn naive_neg_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        sum += a[index] * b[index];
    }
    -sum
}

#[inline(never)]
pub fn naive_squared_l2_f16(a: &[crate::F16], b: &[crate::F16]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        let difference = a[index].to_f32() - b[index].to_f32();
        sum += difference * difference;
    }
    sum
}

#[inline(never)]
pub fn naive_neg_dot_f16(a: &[crate::F16], b: &[crate::F16]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        sum += a[index].to_f32() * b[index].to_f32();
    }
    -sum
}

#[inline(never)]
#[allow(clippy::cast_precision_loss)] // The public i8 contract converts once after accumulation.
pub fn naive_squared_l2_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut sum = 0i32;
    for index in 0..a.len() {
        let difference = i32::from(a[index]) - i32::from(b[index]);
        sum += difference * difference;
    }
    sum as f32
}

#[inline(never)]
#[allow(clippy::cast_precision_loss)] // The public i8 contract converts once after accumulation.
pub fn naive_neg_dot_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut sum = 0i32;
    for index in 0..a.len() {
        sum += i32::from(a[index]) * i32::from(b[index]);
    }
    -(sum as f32)
}

#[inline(never)]
pub fn safe_squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut accumulators = [0.0f32; 8];
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for (a_chunk, b_chunk) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((accumulator, &left), &right) in accumulators.iter_mut().zip(a_chunk).zip(b_chunk) {
            let difference = left - right;
            *accumulator += difference * difference;
        }
    }

    let mut sum = accumulators.into_iter().sum::<f32>();
    for (&left, &right) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let difference = left - right;
        sum += difference * difference;
    }
    sum
}

#[inline(never)]
pub fn safe_neg_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut accumulators = [0.0f32; 8];
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for (a_chunk, b_chunk) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((accumulator, &left), &right) in accumulators.iter_mut().zip(a_chunk).zip(b_chunk) {
            *accumulator += left * right;
        }
    }

    let mut sum = accumulators.into_iter().sum::<f32>();
    for (&left, &right) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += left * right;
    }
    -sum
}

#[inline(never)]
pub fn safe_squared_l2_f16(a: &[crate::F16], b: &[crate::F16]) -> f32 {
    let mut accumulators = [0.0f32; 8];
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for (a_chunk, b_chunk) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((accumulator, &left), &right) in accumulators.iter_mut().zip(a_chunk).zip(b_chunk) {
            let difference = left.to_f32() - right.to_f32();
            *accumulator += difference * difference;
        }
    }

    let mut sum = accumulators.into_iter().sum::<f32>();
    for (&left, &right) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let difference = left.to_f32() - right.to_f32();
        sum += difference * difference;
    }
    sum
}

#[inline(never)]
pub fn safe_neg_dot_f16(a: &[crate::F16], b: &[crate::F16]) -> f32 {
    let mut accumulators = [0.0f32; 8];
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for (a_chunk, b_chunk) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((accumulator, &left), &right) in accumulators.iter_mut().zip(a_chunk).zip(b_chunk) {
            *accumulator += left.to_f32() * right.to_f32();
        }
    }

    let mut sum = accumulators.into_iter().sum::<f32>();
    for (&left, &right) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += left.to_f32() * right.to_f32();
    }
    -sum
}

#[inline(never)]
#[allow(clippy::cast_precision_loss)] // The public i8 contract converts once after accumulation.
pub fn safe_squared_l2_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut accumulators = [0i32; 8];
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for (a_chunk, b_chunk) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((accumulator, &left), &right) in accumulators.iter_mut().zip(a_chunk).zip(b_chunk) {
            let difference = i32::from(left) - i32::from(right);
            *accumulator += difference * difference;
        }
    }

    let mut sum = accumulators.into_iter().sum::<i32>();
    for (&left, &right) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        let difference = i32::from(left) - i32::from(right);
        sum += difference * difference;
    }
    sum as f32
}

#[inline(never)]
#[allow(clippy::cast_precision_loss)] // The public i8 contract converts once after accumulation.
pub fn safe_neg_dot_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut accumulators = [0i32; 8];
    let mut a_chunks = a.chunks_exact(8);
    let mut b_chunks = b.chunks_exact(8);
    for (a_chunk, b_chunk) in a_chunks.by_ref().zip(b_chunks.by_ref()) {
        for ((accumulator, &left), &right) in accumulators.iter_mut().zip(a_chunk).zip(b_chunk) {
            *accumulator += i32::from(left) * i32::from(right);
        }
    }

    let mut sum = accumulators.into_iter().sum::<i32>();
    for (&left, &right) in a_chunks.remainder().iter().zip(b_chunks.remainder()) {
        sum += i32::from(left) * i32::from(right);
    }
    -(sum as f32)
}

#[cfg(test)]
mod tests {
    use super::{
        naive_neg_dot_f16, naive_neg_dot_f32, naive_neg_dot_i8, naive_squared_l2_f16,
        naive_squared_l2_f32, naive_squared_l2_i8, safe_neg_dot_f16, safe_neg_dot_f32,
        safe_neg_dot_i8, safe_squared_l2_f16, safe_squared_l2_f32, safe_squared_l2_i8,
    };
    use crate::{F16, KernelPath, MetricType, ScoreKernel};

    fn assert_bits_equal(actual: f32, expected: f32) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }

    #[test]
    fn baseline_families_equal_public_scalar_path_on_integer_values() {
        let a_f32 = [1.0, 2.0, 3.0, -4.0, 5.0, -6.0, 7.0, 8.0, -9.0, 10.0, 11.0];
        let b_f32 = [4.0, 6.0, 8.0, 2.0, -3.0, 1.0, 5.0, -7.0, 9.0, 0.0, -2.0];
        let a_f16 = a_f32.map(F16::from_f32);
        let b_f16 = b_f32.map(F16::from_f32);
        let a_i8 = [1, 2, 3, -4, 5, -6, 7, 8, -9, 10, 11];
        let b_i8 = [4, 6, 8, 2, -3, 1, 5, -7, 9, 0, -2];

        let l2_f32 = ScoreKernel::<f32>::with_path(MetricType::L2, KernelPath::Scalar).unwrap();
        let dot_f32 =
            ScoreKernel::<f32>::with_path(MetricType::InnerProduct, KernelPath::Scalar).unwrap();
        let l2_f16 = ScoreKernel::<F16>::with_path(MetricType::L2, KernelPath::Scalar).unwrap();
        let dot_f16 =
            ScoreKernel::<F16>::with_path(MetricType::InnerProduct, KernelPath::Scalar).unwrap();
        let l2_i8 = ScoreKernel::<i8>::with_path(MetricType::L2, KernelPath::Scalar).unwrap();
        let dot_i8 =
            ScoreKernel::<i8>::with_path(MetricType::InnerProduct, KernelPath::Scalar).unwrap();

        for score in [
            naive_squared_l2_f32(&a_f32, &b_f32),
            safe_squared_l2_f32(&a_f32, &b_f32),
        ] {
            assert_bits_equal(score, l2_f32.score(&a_f32, &b_f32));
        }
        for score in [
            naive_neg_dot_f32(&a_f32, &b_f32),
            safe_neg_dot_f32(&a_f32, &b_f32),
        ] {
            assert_bits_equal(score, dot_f32.score(&a_f32, &b_f32));
        }
        for score in [
            naive_squared_l2_f16(&a_f16, &b_f16),
            safe_squared_l2_f16(&a_f16, &b_f16),
        ] {
            assert_bits_equal(score, l2_f16.score(&a_f16, &b_f16));
        }
        for score in [
            naive_neg_dot_f16(&a_f16, &b_f16),
            safe_neg_dot_f16(&a_f16, &b_f16),
        ] {
            assert_bits_equal(score, dot_f16.score(&a_f16, &b_f16));
        }
        for score in [
            naive_squared_l2_i8(&a_i8, &b_i8),
            safe_squared_l2_i8(&a_i8, &b_i8),
        ] {
            assert_bits_equal(score, l2_i8.score(&a_i8, &b_i8));
        }
        for score in [
            naive_neg_dot_i8(&a_i8, &b_i8),
            safe_neg_dot_i8(&a_i8, &b_i8),
        ] {
            assert_bits_equal(score, dot_i8.score(&a_i8, &b_i8));
        }
    }
}
