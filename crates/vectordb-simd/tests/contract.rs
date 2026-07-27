#![allow(clippy::float_cmp)] // exact values are the contract under test
#![allow(clippy::should_panic_without_expect)] // panic occurrence is the contract under test

use vectordb_simd::{
    Element, Error, F16, KernelPath, MAX_I8_DIMENSION, MetricType, ScoreKernel, l2_norm,
    normalize_l2,
};

#[test]
fn new_always_resolves_a_path() {
    let kernel = ScoreKernel::<f32>::new(MetricType::L2);
    assert_eq!(kernel.metric(), MetricType::L2);
    assert_eq!(kernel.score(&[0.0, 3.0], &[4.0, 7.0]), 32.0);
}

#[test]
fn scalar_path_is_always_constructible_for_every_element_and_metric() {
    for metric in [MetricType::L2, MetricType::InnerProduct, MetricType::Cosine] {
        assert_eq!(
            ScoreKernel::<f32>::with_path(metric, KernelPath::Scalar)
                .unwrap()
                .path(),
            KernelPath::Scalar
        );
        assert_eq!(
            ScoreKernel::<F16>::with_path(metric, KernelPath::Scalar)
                .unwrap()
                .path(),
            KernelPath::Scalar
        );
        assert_eq!(
            ScoreKernel::<i8>::with_path(metric, KernelPath::Scalar)
                .unwrap()
                .path(),
            KernelPath::Scalar
        );
    }
}

#[test]
fn foreign_architecture_path_is_unsupported() {
    let path = if cfg!(target_arch = "x86_64") {
        KernelPath::Neon
    } else {
        KernelPath::Avx2
    };
    let error = ScoreKernel::<f32>::with_path(MetricType::L2, path).unwrap_err();
    assert!(matches!(error, Error::Unsupported { .. }));
}

#[test]
#[should_panic]
fn score_panics_on_dimension_mismatch() {
    ScoreKernel::<f32>::new(MetricType::L2).score(&[1.0], &[1.0, 2.0]);
}

#[test]
#[should_panic]
fn score_many_panics_on_ragged_target() {
    let mut out = [0.0f32; 2];
    ScoreKernel::<f32>::new(MetricType::L2).score_many(
        &[1.0, 2.0],
        &[&[1.0, 2.0], &[1.0]],
        &mut out,
    );
}

#[test]
#[should_panic]
fn score_many_panics_on_output_length_mismatch() {
    let mut out = [0.0f32; 1];
    ScoreKernel::<f32>::new(MetricType::L2).score_many(&[1.0], &[&[1.0], &[2.0]], &mut out);
}

#[test]
#[should_panic]
fn score_contiguous_panics_on_empty_query() {
    let mut out = [0.0f32; 3];
    ScoreKernel::<f32>::new(MetricType::L2).score_contiguous(&[], &[], &mut out);
}

#[test]
#[should_panic]
fn score_contiguous_panics_on_row_count_mismatch() {
    let mut out = [0.0f32; 2];
    ScoreKernel::<f32>::new(MetricType::L2).score_contiguous(
        &[1.0, 2.0],
        &[1.0, 0.0, 0.0],
        &mut out,
    );
}

#[test]
#[should_panic]
fn i8_kernel_panics_beyond_max_dimension() {
    let v = vec![0i8; MAX_I8_DIMENSION + 1];
    ScoreKernel::<i8>::new(MetricType::L2).score(&v, &v);
}

#[test]
#[should_panic]
fn i8_score_many_panics_beyond_max_dimension() {
    let v = vec![0i8; MAX_I8_DIMENSION + 1];
    let mut out = [0.0f32; 1];
    ScoreKernel::<i8>::new(MetricType::L2).score_many(&v, &[&v], &mut out);
}

#[test]
#[should_panic]
fn i8_score_contiguous_panics_beyond_max_dimension() {
    let v = vec![0i8; MAX_I8_DIMENSION + 1];
    let mut out = [0.0f32; 1];
    ScoreKernel::<i8>::new(MetricType::L2).score_contiguous(&v, &v, &mut out);
}

#[test]
fn i8_kernel_accepts_the_exact_max_dimension() {
    let v = vec![1i8; MAX_I8_DIMENSION];
    assert_eq!(ScoreKernel::<i8>::new(MetricType::L2).score(&v, &v), 0.0);
}

#[test]
fn score_many_scores_each_target() {
    let kernel = ScoreKernel::<f32>::new(MetricType::L2);
    let mut out = [0.0f32; 2];
    kernel.score_many(&[0.0, 0.0], &[&[3.0, 4.0], &[1.0, 0.0]], &mut out);
    assert_eq!(out, [25.0, 1.0]);
}

#[test]
fn batch_validation_precedes_any_output_write() {
    // A ragged SECOND target must leave out untouched, including out[0].
    let kernel = ScoreKernel::<f32>::new(MetricType::L2);
    let mut out = [f32::NAN; 2];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kernel.score_many(&[1.0, 2.0], &[&[1.0, 2.0], &[1.0]], &mut out);
    }));
    assert!(result.is_err());
    assert!(out.iter().all(|x| x.is_nan())); // sentinel values preserved
}

#[test]
fn output_count_validation_precedes_any_output_write() {
    let kernel = ScoreKernel::<f32>::new(MetricType::L2);
    let mut out = [f32::NAN; 2];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        kernel.score_many(&[1.0], &[&[1.0]], &mut out);
    }));
    assert!(result.is_err());
    assert!(out.iter().all(|value| value.is_nan()));
}

#[test]
fn score_contiguous_scores_row_major_vectors() {
    let kernel = ScoreKernel::<f32>::new(MetricType::InnerProduct);
    let mut out = [0.0f32; 2];
    kernel.score_contiguous(&[1.0, 2.0], &[1.0, 0.0, 0.0, 1.0], &mut out);
    assert_eq!(out, [-1.0, -2.0]);
}

#[test]
fn zero_dimension_signed_identity_through_public_api() {
    let l2 = ScoreKernel::<f32>::new(MetricType::L2);
    let ip = ScoreKernel::<f32>::new(MetricType::InnerProduct);
    assert_eq!(l2.score(&[], &[]).to_bits(), 0.0f32.to_bits());
    assert_eq!(ip.score(&[], &[]).to_bits(), (-0.0f32).to_bits());
}

#[test]
fn prefetch_is_callable_on_every_kernel() {
    ScoreKernel::<f32>::new(MetricType::L2).prefetch(&[1.0, 2.0]);
    ScoreKernel::<i8>::new(MetricType::L2).prefetch(&[1, 2]);
}

#[test]
fn normalize_l2_returns_f64_norm_and_unit_result() {
    let mut v = [3.0f32, 4.0];
    assert_eq!(normalize_l2(&mut v), 5.0f64);
    assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
}

#[test]
fn normalize_l2_handles_norm_beyond_f32_range() {
    let mut v = [f32::MAX, f32::MAX];
    let norm = normalize_l2(&mut v);
    assert!(norm > f64::from(f32::MAX));
    assert!(v.iter().all(|x| x.is_finite()));
    assert!((l2_norm(&v) - 1.0).abs() < 1e-6);
}

#[test]
fn normalize_l2_leaves_zero_vector_unchanged() {
    let mut v = [0.0f32; 3];
    assert_eq!(normalize_l2(&mut v), 0.0f64);
    assert_eq!(v, [0.0; 3]);
}

#[test]
fn kernel_debug_names_metric_and_path() {
    let text = format!("{:?}", ScoreKernel::<f32>::new(MetricType::L2));
    assert!(text.contains("L2"));
    assert!(!text.contains("0x")); // no pointer addresses
}

#[test]
fn kernels_are_send_sync_copy() {
    fn assert_bounds<T: Send + Sync + Copy>() {}
    assert_bounds::<ScoreKernel<f32>>();
    assert_bounds::<ScoreKernel<F16>>();
    assert_bounds::<ScoreKernel<i8>>();
}

fn assert_cosine_matches_inner_product<T: Element>(a: &[T], b: &[T]) {
    let cosine = ScoreKernel::<T>::new(MetricType::Cosine);
    let inner_product = ScoreKernel::<T>::new(MetricType::InnerProduct);
    assert_eq!(cosine.path(), inner_product.path());
    assert_eq!(
        cosine.score(a, b).to_bits(),
        inner_product.score(a, b).to_bits()
    );
}

#[test]
fn cosine_and_inner_product_share_score_kernels() {
    assert_cosine_matches_inner_product(&[1.0f32, -2.0, 3.0], &[-4.0, 5.0, 6.0]);
    assert_cosine_matches_inner_product(
        &[F16::from_f32(1.0), F16::from_f32(-2.0), F16::from_f32(3.0)],
        &[F16::from_f32(-4.0), F16::from_f32(5.0), F16::from_f32(6.0)],
    );
    assert_cosine_matches_inner_product(&[1i8, -2, 3], &[-4, 5, 6]);
}
