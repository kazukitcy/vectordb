use std::hint::black_box;

#[inline(never)]
pub fn squared_l2_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        let difference = a[index] - b[index];
        sum = black_box(sum + difference * difference);
    }
    sum
}

#[inline(never)]
pub fn neg_dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        sum = black_box(sum + a[index] * b[index]);
    }
    -sum
}

#[inline(never)]
pub fn squared_l2_f16(a: &[crate::F16], b: &[crate::F16]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        let difference = a[index].to_f32() - b[index].to_f32();
        sum = black_box(sum + difference * difference);
    }
    sum
}

#[inline(never)]
pub fn neg_dot_f16(a: &[crate::F16], b: &[crate::F16]) -> f32 {
    let mut sum = 0.0;
    for index in 0..a.len() {
        sum = black_box(sum + a[index].to_f32() * b[index].to_f32());
    }
    -sum
}

#[inline(never)]
#[allow(clippy::cast_precision_loss)]
pub fn squared_l2_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut sum = 0i32;
    for index in 0..a.len() {
        let difference = i32::from(a[index]) - i32::from(b[index]);
        sum = black_box(sum + difference * difference);
    }
    sum as f32
}

#[inline(never)]
#[allow(clippy::cast_precision_loss)]
pub fn neg_dot_i8(a: &[i8], b: &[i8]) -> f32 {
    let mut sum = 0i32;
    for index in 0..a.len() {
        sum = black_box(sum + i32::from(a[index]) * i32::from(b[index]));
    }
    -(sum as f32)
}

#[cfg(test)]
// Exact values are the contract under test.
#[allow(clippy::float_cmp)]
mod tests {
    use super::{
        neg_dot_f16, neg_dot_f32, neg_dot_i8, squared_l2_f16, squared_l2_f32, squared_l2_i8,
    };
    use crate::{F16, KernelPath, MetricType, ScoreKernel};

    #[test]
    fn baseline_values_equal_public_scalar_path_values() {
        let a_f32 = [1.0, 2.0, 3.0];
        let b_f32 = [4.0, 6.0, 8.0];
        let a_f16 = a_f32.map(F16::from_f32);
        let b_f16 = b_f32.map(F16::from_f32);
        let a_i8 = [1, 2, 3];
        let b_i8 = [4, 6, 8];

        let l2_f32 = ScoreKernel::<f32>::with_path(MetricType::L2, KernelPath::Scalar).unwrap();
        let dot_f32 =
            ScoreKernel::<f32>::with_path(MetricType::InnerProduct, KernelPath::Scalar).unwrap();
        let l2_f16 = ScoreKernel::<F16>::with_path(MetricType::L2, KernelPath::Scalar).unwrap();
        let dot_f16 =
            ScoreKernel::<F16>::with_path(MetricType::InnerProduct, KernelPath::Scalar).unwrap();
        let l2_i8 = ScoreKernel::<i8>::with_path(MetricType::L2, KernelPath::Scalar).unwrap();
        let dot_i8 =
            ScoreKernel::<i8>::with_path(MetricType::InnerProduct, KernelPath::Scalar).unwrap();

        assert_eq!(squared_l2_f32(&a_f32, &b_f32), l2_f32.score(&a_f32, &b_f32));
        assert_eq!(neg_dot_f32(&a_f32, &b_f32), dot_f32.score(&a_f32, &b_f32));
        assert_eq!(squared_l2_f16(&a_f16, &b_f16), l2_f16.score(&a_f16, &b_f16));
        assert_eq!(neg_dot_f16(&a_f16, &b_f16), dot_f16.score(&a_f16, &b_f16));
        assert_eq!(squared_l2_i8(&a_i8, &b_i8), l2_i8.score(&a_i8, &b_i8));
        assert_eq!(neg_dot_i8(&a_i8, &b_i8), dot_i8.score(&a_i8, &b_i8));
    }
}
