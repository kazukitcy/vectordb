#![allow(clippy::float_cmp)] // exact values are the contract under test

use std::cell::Cell;
use std::collections::HashSet;

use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestRunner};
use vectordb_simd::{F16, KernelPath, MetricType, ScoreKernel};

const ALL_PATHS: [KernelPath; 4] = [
    KernelPath::Scalar,
    KernelPath::Avx2,
    KernelPath::Avx512,
    KernelPath::Neon,
];
const METRICS: [MetricType; 3] = [MetricType::L2, MetricType::InnerProduct, MetricType::Cosine];
const FIXED_DIMENSIONS: [usize; 21] = [
    1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 4095, 4096,
];
const OFFSETS: [usize; 3] = [0, 1, 3];
const EPSILON: f64 = f32::EPSILON as f64;

type CheckResult = Result<(), String>;

struct PathRun {
    path: KernelPath,
    oracle_a_cases: Cell<usize>,
    oracle_b_cases: Cell<usize>,
}

impl PathRun {
    const fn new(path: KernelPath) -> Self {
        Self {
            path,
            oracle_a_cases: Cell::new(0),
            oracle_b_cases: Cell::new(0),
        }
    }

    fn count_oracle_a(&self) {
        self.oracle_a_cases.set(self.oracle_a_cases.get() + 1);
    }

    fn count_oracle_b(&self) {
        self.oracle_b_cases.set(self.oracle_b_cases.get() + 1);
    }
}

fn required_paths() -> HashSet<KernelPath> {
    let value = std::env::var("VECTORDB_SIMD_REQUIRE").unwrap_or_default();
    value
        .split(',')
        .filter_map(|name| {
            let name = name.trim();
            if name.is_empty() {
                return None;
            }
            Some(match name.to_ascii_lowercase().as_str() {
                "scalar" => KernelPath::Scalar,
                "avx2" => KernelPath::Avx2,
                "avx512" => KernelPath::Avx512,
                "neon" => KernelPath::Neon,
                _ => panic!("unknown VECTORDB_SIMD_REQUIRE path: {name}"),
            })
        })
        .collect()
}

fn construction_failures(path: KernelPath) -> Vec<String> {
    let mut failures = Vec::new();
    for metric in METRICS {
        if let Err(error) = ScoreKernel::<f32>::with_path(metric, path) {
            failures.push(format!("f32/{metric:?}: {error}"));
        }
        if let Err(error) = ScoreKernel::<F16>::with_path(metric, path) {
            failures.push(format!("f16/{metric:?}: {error}"));
        }
        if let Err(error) = ScoreKernel::<i8>::with_path(metric, path) {
            failures.push(format!("i8/{metric:?}: {error}"));
        }
    }
    failures
}

fn constructible_paths() -> Vec<PathRun> {
    let required = required_paths();
    let mut paths = Vec::new();

    for path in ALL_PATHS {
        let failures = construction_failures(path);
        if failures.is_empty() {
            paths.push(PathRun::new(path));
        } else if required.contains(&path) {
            panic!(
                "required path {path:?} is unavailable:\n{}",
                failures.join("\n")
            );
        } else {
            println!("path skipped: {path:?}");
        }
    }

    paths
}

fn f64_oracle<T: Copy>(
    metric: MetricType,
    a: &[T],
    b: &[T],
    to_f64: impl Fn(T) -> f64,
) -> (f64, f64) {
    match metric {
        MetricType::L2 => {
            let mut score = 0.0;
            let mut term_magnitudes = 0.0;
            for (&left, &right) in a.iter().zip(b) {
                let difference = to_f64(left) - to_f64(right);
                let term = difference * difference;
                score += term;
                term_magnitudes += term.abs();
            }
            (score, term_magnitudes)
        }
        MetricType::InnerProduct | MetricType::Cosine => {
            let mut dot = 0.0;
            let mut term_magnitudes = 0.0;
            for (&left, &right) in a.iter().zip(b) {
                let term = to_f64(left) * to_f64(right);
                dot += term;
                term_magnitudes += term.abs();
            }
            (-dot, term_magnitudes)
        }
        _ => unreachable!("unknown metric variant"),
    }
}

fn exact_bits(
    path: KernelPath,
    element: &str,
    metric: MetricType,
    got: f32,
    scalar: f32,
    expected: f32,
) -> CheckResult {
    if got.to_bits() != scalar.to_bits()
        || got.to_bits() != expected.to_bits()
        || scalar.to_bits() != expected.to_bits()
    {
        return Err(format!(
            "{element}/{metric:?}/{path:?}: got={got:?} ({:#010x}), \
             scalar={scalar:?} ({:#010x}), f64-oracle={expected:?} ({:#010x})",
            got.to_bits(),
            scalar.to_bits(),
            expected.to_bits()
        ));
    }
    Ok(())
}

// The cast is the specified final step of the independent f64 oracle.
#[allow(clippy::cast_possible_truncation)]
fn assert_exact_f32(path: KernelPath, metric: MetricType, a: &[f32], b: &[f32]) -> CheckResult {
    let got = ScoreKernel::<f32>::with_path(metric, path)
        .expect("constructible path")
        .score(a, b);
    let scalar = ScoreKernel::<f32>::with_path(metric, KernelPath::Scalar)
        .expect("scalar path")
        .score(a, b);
    let expected = f64_oracle(metric, a, b, f64::from).0 as f32;
    exact_bits(path, "f32", metric, got, scalar, expected)
}

// The cast is the specified final step of the independent f64 oracle.
#[allow(clippy::cast_possible_truncation)]
fn assert_exact_f16(path: KernelPath, metric: MetricType, a: &[F16], b: &[F16]) -> CheckResult {
    let got = ScoreKernel::<F16>::with_path(metric, path)
        .expect("constructible path")
        .score(a, b);
    let scalar = ScoreKernel::<F16>::with_path(metric, KernelPath::Scalar)
        .expect("scalar path")
        .score(a, b);
    let expected = f64_oracle(metric, a, b, |value| f64::from(value.to_f32())).0 as f32;
    exact_bits(path, "f16", metric, got, scalar, expected)
}

// The cast pins the kernel's one final exact-integer-to-f32 conversion.
#[allow(clippy::cast_possible_truncation)]
fn assert_exact_i8(path: KernelPath, metric: MetricType, a: &[i8], b: &[i8]) -> CheckResult {
    let got = ScoreKernel::<i8>::with_path(metric, path)
        .expect("constructible path")
        .score(a, b);
    let scalar = ScoreKernel::<i8>::with_path(metric, KernelPath::Scalar)
        .expect("scalar path")
        .score(a, b);
    let expected = f64_oracle(metric, a, b, f64::from).0 as f32;
    exact_bits(path, "i8", metric, got, scalar, expected)
}

fn tolerance(metric: MetricType, a_len: usize, term_magnitudes: f64) -> f64 {
    let dimension = f64::from(u32::try_from(a_len).expect("test dimension fits in u32"));
    let _ = metric;
    (4.0 * dimension * EPSILON * term_magnitudes).max(1e-6)
}

fn within_tolerance(
    path: KernelPath,
    element: &str,
    metric: MetricType,
    got: f32,
    scalar: f32,
    expected: f64,
    allowed: f64,
) -> CheckResult {
    let got_error = (f64::from(got) - expected).abs();
    let scalar_error = (f64::from(scalar) - expected).abs();
    if got_error > allowed || scalar_error > allowed {
        return Err(format!(
            "{element}/{metric:?}/{path:?}: got={got:?}, scalar={scalar:?}, \
             f64-oracle={expected:?}, got-error={got_error:?}, \
             scalar-error={scalar_error:?}, tolerance={allowed:?}"
        ));
    }
    Ok(())
}

fn assert_tolerant_f32(path: KernelPath, metric: MetricType, a: &[f32], b: &[f32]) -> CheckResult {
    let got = ScoreKernel::<f32>::with_path(metric, path)
        .expect("constructible path")
        .score(a, b);
    let scalar = ScoreKernel::<f32>::with_path(metric, KernelPath::Scalar)
        .expect("scalar path")
        .score(a, b);
    let (expected, term_magnitudes) = f64_oracle(metric, a, b, f64::from);
    within_tolerance(
        path,
        "f32",
        metric,
        got,
        scalar,
        expected,
        tolerance(metric, a.len(), term_magnitudes),
    )
}

fn assert_tolerant_f16(path: KernelPath, metric: MetricType, a: &[F16], b: &[F16]) -> CheckResult {
    let got = ScoreKernel::<F16>::with_path(metric, path)
        .expect("constructible path")
        .score(a, b);
    let scalar = ScoreKernel::<F16>::with_path(metric, KernelPath::Scalar)
        .expect("scalar path")
        .score(a, b);
    let (expected, term_magnitudes) = f64_oracle(metric, a, b, |value| f64::from(value.to_f32()));
    within_tolerance(
        path,
        "f16",
        metric,
        got,
        scalar,
        expected,
        tolerance(metric, a.len(), term_magnitudes),
    )
}

fn with_offsets<T: Copy>(
    a: &[T],
    b: &[T],
    mut check: impl FnMut(&[T], &[T]) -> CheckResult,
) -> CheckResult {
    for offset in OFFSETS {
        let mut a_storage = vec![a[0]; offset];
        a_storage.extend_from_slice(a);
        let mut b_storage = vec![b[0]; offset];
        b_storage.extend_from_slice(b);
        check(&a_storage[offset..], &b_storage[offset..])?;
    }
    Ok(())
}

fn run_oracle_a_case(
    paths: &[PathRun],
    grid_a: &[i8],
    grid_b: &[i8],
    i8_a: &[i8],
    i8_b: &[i8],
) -> CheckResult {
    let f32_a: Vec<f32> = grid_a.iter().copied().map(f32::from).collect();
    let f32_b: Vec<f32> = grid_b.iter().copied().map(f32::from).collect();
    let f16_a: Vec<F16> = f32_a.iter().copied().map(F16::from_f32).collect();
    let f16_b: Vec<F16> = f32_b.iter().copied().map(F16::from_f32).collect();

    with_offsets(&f32_a, &f32_b, |a, b| {
        for path in paths {
            for metric in METRICS {
                assert_exact_f32(path.path, metric, a, b)?;
                path.count_oracle_a();
            }
        }
        Ok(())
    })?;
    with_offsets(&f16_a, &f16_b, |a, b| {
        for path in paths {
            for metric in METRICS {
                assert_exact_f16(path.path, metric, a, b)?;
                path.count_oracle_a();
            }
        }
        Ok(())
    })?;
    with_offsets(i8_a, i8_b, |a, b| {
        for path in paths {
            for metric in METRICS {
                assert_exact_i8(path.path, metric, a, b)?;
                path.count_oracle_a();
            }
        }
        Ok(())
    })
}

fn run_oracle_b_case(
    paths: &[PathRun],
    f32_a: &[f32],
    f32_b: &[f32],
    i8_a: &[i8],
    i8_b: &[i8],
) -> CheckResult {
    let f16_a: Vec<F16> = f32_a.iter().copied().map(F16::from_f32).collect();
    let f16_b: Vec<F16> = f32_b.iter().copied().map(F16::from_f32).collect();

    with_offsets(f32_a, f32_b, |a, b| {
        for path in paths {
            for metric in METRICS {
                assert_tolerant_f32(path.path, metric, a, b)?;
                path.count_oracle_b();
            }
        }
        Ok(())
    })?;
    with_offsets(&f16_a, &f16_b, |a, b| {
        for path in paths {
            for metric in METRICS {
                assert_tolerant_f16(path.path, metric, a, b)?;
                path.count_oracle_b();
            }
        }
        Ok(())
    })?;
    with_offsets(i8_a, i8_b, |a, b| {
        for path in paths {
            for metric in METRICS {
                assert_exact_i8(path.path, metric, a, b)?;
                path.count_oracle_b();
            }
        }
        Ok(())
    })
}

fn fixed_grid(dimension: usize, multiplier: usize, addend: usize) -> Vec<i8> {
    (0..dimension)
        .map(|index| {
            let value = (index * multiplier + addend) % 33;
            i8::try_from(value).expect("grid value fits in i8") - 16
        })
        .collect()
}

fn fixed_i8(dimension: usize, multiplier: usize, addend: usize) -> Vec<i8> {
    (0..dimension)
        .map(|index| {
            let byte = u8::try_from((index * multiplier + addend) % 256).expect("byte value fits");
            i8::from_ne_bytes([byte])
        })
        .collect()
}

fn run_fixed_dimensions(paths: &[PathRun]) {
    for dimension in FIXED_DIMENSIONS {
        run_oracle_a_case(
            paths,
            &fixed_grid(dimension, 17, 5),
            &fixed_grid(dimension, 29, 11),
            &fixed_i8(dimension, 73, 19),
            &fixed_i8(dimension, 151, 47),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    }
}

fn run_zero_and_cancellation_cases(paths: &[PathRun]) {
    let zero_f32 = [0.0f32; 17];
    let zero_f16 = [F16::from_f32(0.0); 17];
    let zero_i8 = [0i8; 17];

    for path in paths {
        for metric in METRICS {
            assert_exact_f32(path.path, metric, &zero_f32, &zero_f32)
                .unwrap_or_else(|error| panic!("{error}"));
            assert_exact_f16(path.path, metric, &zero_f16, &zero_f16)
                .unwrap_or_else(|error| panic!("{error}"));
            assert_exact_i8(path.path, metric, &zero_i8, &zero_i8)
                .unwrap_or_else(|error| panic!("{error}"));
            path.count_oracle_a();
            path.count_oracle_a();
            path.count_oracle_a();
        }

        for metric in [MetricType::InnerProduct, MetricType::Cosine] {
            assert_exact_f32(path.path, metric, &[1.0, 1.0], &[1.0, -1.0])
                .unwrap_or_else(|error| panic!("{error}"));
            assert_exact_f16(
                path.path,
                metric,
                &[F16::from_f32(1.0), F16::from_f32(1.0)],
                &[F16::from_f32(1.0), F16::from_f32(-1.0)],
            )
            .unwrap_or_else(|error| panic!("{error}"));
            assert_exact_i8(path.path, metric, &[1, 1], &[1, -1])
                .unwrap_or_else(|error| panic!("{error}"));
            path.count_oracle_a();
            path.count_oracle_a();
            path.count_oracle_a();
        }

        for metric in METRICS {
            assert_exact_f32(path.path, metric, &[], &[]).unwrap_or_else(|error| panic!("{error}"));
            assert_exact_f16(path.path, metric, &[], &[]).unwrap_or_else(|error| panic!("{error}"));
            assert_exact_i8(path.path, metric, &[], &[]).unwrap_or_else(|error| panic!("{error}"));
            path.count_oracle_a();
            path.count_oracle_a();
            path.count_oracle_a();
        }
    }
}

fn i8_l2_pair(score: u32) -> (Vec<i8>, Vec<i8>) {
    let mut remaining = score;
    let mut a = Vec::new();
    let mut b = Vec::new();

    while remaining != 0 {
        let mut difference = remaining.min(255);
        while difference * difference > remaining {
            difference -= 1;
        }
        let left = difference.min(127);
        let right = i32::try_from(left).expect("left fits")
            - i32::try_from(difference).expect("difference fits");
        a.push(i8::try_from(left).expect("left fits in i8"));
        b.push(i8::try_from(right).expect("right fits in i8"));
        remaining -= difference * difference;
    }

    (a, b)
}

fn i8_neg_dot_pair(score: u32) -> (Vec<i8>, Vec<i8>) {
    const LARGE_TERM: u32 = 127 * 128;
    let mut remaining = score;
    let mut a = Vec::new();
    let mut b = Vec::new();

    let large_terms = remaining / LARGE_TERM;
    for _ in 0..large_terms {
        a.push(127);
        b.push(-128);
    }
    remaining %= LARGE_TERM;

    let terms_128 = remaining / 128;
    for _ in 0..terms_128 {
        a.push(1);
        b.push(-128);
    }
    remaining %= 128;

    if remaining != 0 {
        a.push(1);
        b.push(-i8::try_from(remaining).expect("remainder fits in i8"));
    }

    (a, b)
}

fn run_i8_conversion_boundaries(paths: &[PathRun]) {
    for score in [(1_u32 << 24) - 1, 1_u32 << 24, (1_u32 << 24) + 1] {
        let (l2_a, l2_b) = i8_l2_pair(score);
        let (dot_a, dot_b) = i8_neg_dot_pair(score);
        for path in paths {
            assert_exact_i8(path.path, MetricType::L2, &l2_a, &l2_b)
                .unwrap_or_else(|error| panic!("{error}"));
            path.count_oracle_a();
            for metric in [MetricType::InnerProduct, MetricType::Cosine] {
                assert_exact_i8(path.path, metric, &dot_a, &dot_b)
                    .unwrap_or_else(|error| panic!("{error}"));
                path.count_oracle_a();
            }
        }
    }
}

fn oracle_a_strategy() -> impl Strategy<Value = (Vec<i8>, Vec<i8>, Vec<i8>, Vec<i8>)> {
    (1_usize..=4096).prop_flat_map(|dimension| {
        (
            prop::collection::vec(-16_i8..=16, dimension),
            prop::collection::vec(-16_i8..=16, dimension),
            prop::collection::vec(any::<i8>(), dimension),
            prop::collection::vec(any::<i8>(), dimension),
        )
    })
}

fn oracle_b_strategy() -> impl Strategy<Value = (Vec<f32>, Vec<f32>, Vec<i8>, Vec<i8>)> {
    (1_usize..=4096).prop_flat_map(|dimension| {
        (
            prop::collection::vec(-100.0_f32..100.0, dimension),
            prop::collection::vec(-100.0_f32..100.0, dimension),
            prop::collection::vec(any::<i8>(), dimension),
            prop::collection::vec(any::<i8>(), dimension),
        )
    })
}

fn property_runner() -> TestRunner {
    TestRunner::new(ProptestConfig {
        cases: 64,
        failure_persistence: None,
        ..ProptestConfig::default()
    })
}

#[test]
fn all_constructible_paths_match_dual_oracles() {
    let paths = constructible_paths();
    assert!(!paths.is_empty(), "the scalar path must be constructible");

    run_fixed_dimensions(&paths);
    run_zero_and_cancellation_cases(&paths);
    run_i8_conversion_boundaries(&paths);

    property_runner()
        .run(&oracle_a_strategy(), |(grid_a, grid_b, i8_a, i8_b)| {
            run_oracle_a_case(&paths, &grid_a, &grid_b, &i8_a, &i8_b).map_err(TestCaseError::fail)
        })
        .unwrap();

    property_runner()
        .run(&oracle_b_strategy(), |(f32_a, f32_b, i8_a, i8_b)| {
            run_oracle_b_case(&paths, &f32_a, &f32_b, &i8_a, &i8_b).map_err(TestCaseError::fail)
        })
        .unwrap();

    for path in &paths {
        assert!(path.oracle_a_cases.get() > 0);
        assert!(path.oracle_b_cases.get() > 0);
        println!(
            "path exercised: {:?}; oracle A cases: {}; oracle B cases: {}",
            path.path,
            path.oracle_a_cases.get(),
            path.oracle_b_cases.get()
        );
    }
}
