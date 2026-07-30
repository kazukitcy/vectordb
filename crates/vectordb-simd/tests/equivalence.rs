use std::cell::Cell;

use proptest::prelude::*;
use proptest::test_runner::{Config as ProptestConfig, TestCaseError, TestRunner};
use vectordb_simd::{Element, F16, KernelPath, MAX_I8_DIMENSION, MetricType, ScoreKernel};

const ALL_PATHS: [KernelPath; 4] = [
    KernelPath::Scalar,
    KernelPath::Avx2,
    KernelPath::Avx512,
    KernelPath::Neon,
];
const SCORE_METRICS: [MetricType; 2] = [MetricType::L2, MetricType::InnerProduct];
const BATCH_DIMENSIONS: [usize; 5] = [1, 63, 64, 65, 4095];
const FIXED_DIMENSIONS: [usize; 21] = [
    1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 4095, 4096,
];
const OFFSETS: [usize; 3] = [0, 1, 3];
const EPSILON: f64 = f32::EPSILON as f64;
const DPBUSD_BYTE_LANES: usize = 64;
const DPBUSD_I32_LANES: usize = 16;

type CheckResult = Result<(), String>;

fn emulate_dpbusd(
    mut accumulators: [i32; DPBUSD_I32_LANES],
    unsigned: &[u8],
    signed: &[i8],
) -> [i32; DPBUSD_I32_LANES] {
    assert_eq!(unsigned.len(), DPBUSD_BYTE_LANES);
    assert_eq!(signed.len(), DPBUSD_BYTE_LANES);

    for (lane, accumulator) in accumulators.iter_mut().enumerate() {
        let start = lane * 4;
        for element in 0..4 {
            *accumulator +=
                i32::from(unsigned[start + element]) * i32::from(signed[start + element]);
        }
    }
    accumulators
}

fn vnni_bias_identity_sides(query: &[i8], target: &[i8]) -> (i32, i32) {
    assert_eq!(query.len(), DPBUSD_BYTE_LANES);
    assert_eq!(target.len(), DPBUSD_BYTE_LANES);

    let biased_query: Vec<u8> = query
        .iter()
        .map(|value| u8::from_ne_bytes(value.to_ne_bytes()) ^ 0x80)
        .collect();
    let dpbusd_total: i32 = emulate_dpbusd([0; DPBUSD_I32_LANES], &biased_query, target)
        .into_iter()
        .sum();
    let target_sum: i32 = target.iter().copied().map(i32::from).sum();
    let dot: i32 = query
        .iter()
        .copied()
        .zip(target.iter().copied())
        .map(|(left, right)| i32::from(left) * i32::from(right))
        .sum();
    (dot, dpbusd_total - 128 * target_sum)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TestElement {
    F32,
    F16,
    I8,
}

impl TestElement {
    const fn name(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::I8 => "i8",
        }
    }
}

struct CombinationRun {
    path: KernelPath,
    element: TestElement,
    metric: MetricType,
    oracle_a_cases: Cell<usize>,
    oracle_b_cases: Cell<usize>,
}

impl CombinationRun {
    const fn new(path: KernelPath, element: TestElement, metric: MetricType) -> Self {
        Self {
            path,
            element,
            metric,
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

fn parse_required_path(name: &str) -> KernelPath {
    match name {
        "scalar" => KernelPath::Scalar,
        "avx2" => KernelPath::Avx2,
        "avx512" => KernelPath::Avx512,
        "neon" => KernelPath::Neon,
        _ => panic!("unknown VECTORDB_SIMD_REQUIRE path: {name}"),
    }
}

fn parse_required_element(name: &str) -> TestElement {
    match name {
        "f32" => TestElement::F32,
        "f16" => TestElement::F16,
        "i8" => TestElement::I8,
        _ => panic!("unknown VECTORDB_SIMD_REQUIRE element: {name}"),
    }
}

// Each entry is `path` (every element must construct) or `path:element`
// (only that element must construct on the path).
fn required_combinations() -> Vec<(KernelPath, Option<TestElement>)> {
    let value = std::env::var("VECTORDB_SIMD_REQUIRE").unwrap_or_default();
    value
        .split(',')
        .filter_map(|entry| {
            let entry = entry.trim().to_ascii_lowercase();
            if entry.is_empty() {
                return None;
            }
            Some(match entry.split_once(':') {
                Some((path, element)) => (
                    parse_required_path(path),
                    Some(parse_required_element(element)),
                ),
                None => (parse_required_path(&entry), None),
            })
        })
        .collect()
}

fn construction_error(
    path: KernelPath,
    element: TestElement,
    metric: MetricType,
) -> Option<String> {
    let result = match element {
        TestElement::F32 => ScoreKernel::<f32>::with_path(metric, path).map(|_| ()),
        TestElement::F16 => ScoreKernel::<F16>::with_path(metric, path).map(|_| ()),
        TestElement::I8 => ScoreKernel::<i8>::with_path(metric, path).map(|_| ()),
    };
    result.err().map(|error| error.to_string())
}

fn constructible_combinations() -> Vec<CombinationRun> {
    let required = required_combinations();
    let mut combinations = Vec::new();
    let mut required_failures = Vec::new();

    for path in ALL_PATHS {
        for metric in SCORE_METRICS {
            for element in [TestElement::F32, TestElement::F16, TestElement::I8] {
                if let Some(error) = construction_error(path, element, metric) {
                    println!(
                        "combination skipped: path={path:?} element={} metric={metric:?}: {error}",
                        element.name()
                    );
                    let is_required = required
                        .iter()
                        .any(|(p, e)| *p == path && e.is_none_or(|e| e == element));
                    if is_required {
                        required_failures.push(format!(
                            "path={path:?} element={} metric={metric:?}: {error}",
                            element.name()
                        ));
                    }
                } else {
                    combinations.push(CombinationRun::new(path, element, metric));
                }
            }
        }
    }

    assert!(
        required_failures.is_empty(),
        "required SIMD combinations are unavailable:\n{}",
        required_failures.join("\n")
    );
    combinations
}

fn matching_combinations(
    combinations: &[CombinationRun],
    element: TestElement,
    metric: MetricType,
) -> impl Iterator<Item = &CombinationRun> {
    combinations
        .iter()
        .filter(move |run| run.element == element && run.metric == metric)
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
#[expect(clippy::cast_possible_truncation)]
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
#[expect(clippy::cast_possible_truncation)]
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
#[expect(clippy::cast_possible_truncation)]
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

fn reordering_bound(a_len: usize, term_magnitudes: f64) -> f64 {
    let dimension = f64::from(u32::try_from(a_len).expect("test dimension fits in u32"));
    2.0 * dimension * EPSILON * term_magnitudes
}

fn within_reordering_bound(
    path: KernelPath,
    element: &str,
    metric: MetricType,
    got: f32,
    scalar: f32,
    bound: f64,
) -> CheckResult {
    let difference = (f64::from(got) - f64::from(scalar)).abs();
    if !(difference.is_finite() && difference <= bound) {
        return Err(format!(
            "{element}/{metric:?}/{path:?}: got={got:?}, scalar={scalar:?}, \
             difference={difference:?}, reordering-bound={bound:?}"
        ));
    }
    Ok(())
}

// SIMD and scalar reorder the same terms, so |simd-scalar| <= 2*n*EPS*sum|terms|; Oracle A owns lane accounting.
fn assert_reordered_f32(path: KernelPath, metric: MetricType, a: &[f32], b: &[f32]) -> CheckResult {
    let got = ScoreKernel::<f32>::with_path(metric, path)
        .expect("constructible path")
        .score(a, b);
    let scalar = ScoreKernel::<f32>::with_path(metric, KernelPath::Scalar)
        .expect("scalar path")
        .score(a, b);
    let term_magnitudes = f64_oracle(metric, a, b, f64::from).1;
    within_reordering_bound(
        path,
        "f32",
        metric,
        got,
        scalar,
        reordering_bound(a.len(), term_magnitudes),
    )
}

fn assert_reordered_f16(path: KernelPath, metric: MetricType, a: &[F16], b: &[F16]) -> CheckResult {
    let got = ScoreKernel::<F16>::with_path(metric, path)
        .expect("constructible path")
        .score(a, b);
    let scalar = ScoreKernel::<F16>::with_path(metric, KernelPath::Scalar)
        .expect("scalar path")
        .score(a, b);
    let term_magnitudes = f64_oracle(metric, a, b, |value| f64::from(value.to_f32())).1;
    within_reordering_bound(
        path,
        "f16",
        metric,
        got,
        scalar,
        reordering_bound(a.len(), term_magnitudes),
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
    combinations: &[CombinationRun],
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
        for metric in SCORE_METRICS {
            for run in matching_combinations(combinations, TestElement::F32, metric) {
                assert_exact_f32(run.path, metric, a, b)?;
                run.count_oracle_a();
            }
        }
        Ok(())
    })?;
    with_offsets(&f16_a, &f16_b, |a, b| {
        for metric in SCORE_METRICS {
            for run in matching_combinations(combinations, TestElement::F16, metric) {
                assert_exact_f16(run.path, metric, a, b)?;
                run.count_oracle_a();
            }
        }
        Ok(())
    })?;
    with_offsets(i8_a, i8_b, |a, b| {
        for metric in SCORE_METRICS {
            for run in matching_combinations(combinations, TestElement::I8, metric) {
                assert_exact_i8(run.path, metric, a, b)?;
                run.count_oracle_a();
            }
        }
        Ok(())
    })
}

fn run_oracle_b_case(
    combinations: &[CombinationRun],
    f32_a: &[f32],
    f32_b: &[f32],
    i8_a: &[i8],
    i8_b: &[i8],
) -> CheckResult {
    let f16_a: Vec<F16> = f32_a.iter().copied().map(F16::from_f32).collect();
    let f16_b: Vec<F16> = f32_b.iter().copied().map(F16::from_f32).collect();

    with_offsets(f32_a, f32_b, |a, b| {
        for metric in SCORE_METRICS {
            for run in matching_combinations(combinations, TestElement::F32, metric) {
                assert_reordered_f32(run.path, metric, a, b)?;
                run.count_oracle_b();
            }
        }
        Ok(())
    })?;
    with_offsets(&f16_a, &f16_b, |a, b| {
        for metric in SCORE_METRICS {
            for run in matching_combinations(combinations, TestElement::F16, metric) {
                assert_reordered_f16(run.path, metric, a, b)?;
                run.count_oracle_b();
            }
        }
        Ok(())
    })?;
    with_offsets(i8_a, i8_b, |a, b| {
        for metric in SCORE_METRICS {
            for run in matching_combinations(combinations, TestElement::I8, metric) {
                assert_exact_i8(run.path, metric, a, b)?;
                run.count_oracle_b();
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

fn run_fixed_dimensions(combinations: &[CombinationRun]) {
    for dimension in FIXED_DIMENSIONS {
        run_oracle_a_case(
            combinations,
            &fixed_grid(dimension, 17, 5),
            &fixed_grid(dimension, 29, 11),
            &fixed_i8(dimension, 73, 19),
            &fixed_i8(dimension, 151, 47),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    }
}

fn run_zero_and_cancellation_cases(combinations: &[CombinationRun]) {
    let zero_f32 = [0.0f32; 17];
    let zero_f16 = [F16::from_f32(0.0); 17];
    let zero_i8 = [0i8; 17];

    for metric in SCORE_METRICS {
        for run in matching_combinations(combinations, TestElement::F32, metric) {
            assert_exact_f32(run.path, metric, &zero_f32, &zero_f32)
                .unwrap_or_else(|error| panic!("{error}"));
            run.count_oracle_a();
        }
        for run in matching_combinations(combinations, TestElement::F16, metric) {
            assert_exact_f16(run.path, metric, &zero_f16, &zero_f16)
                .unwrap_or_else(|error| panic!("{error}"));
            run.count_oracle_a();
        }
        for run in matching_combinations(combinations, TestElement::I8, metric) {
            assert_exact_i8(run.path, metric, &zero_i8, &zero_i8)
                .unwrap_or_else(|error| panic!("{error}"));
            run.count_oracle_a();
        }
    }

    for run in matching_combinations(combinations, TestElement::F32, MetricType::InnerProduct) {
        assert_exact_f32(run.path, run.metric, &[1.0, 1.0], &[1.0, -1.0])
            .unwrap_or_else(|error| panic!("{error}"));
        run.count_oracle_a();
    }
    for run in matching_combinations(combinations, TestElement::F16, MetricType::InnerProduct) {
        assert_exact_f16(
            run.path,
            run.metric,
            &[F16::from_f32(1.0), F16::from_f32(1.0)],
            &[F16::from_f32(1.0), F16::from_f32(-1.0)],
        )
        .unwrap_or_else(|error| panic!("{error}"));
        run.count_oracle_a();
    }
    for run in matching_combinations(combinations, TestElement::I8, MetricType::InnerProduct) {
        assert_exact_i8(run.path, run.metric, &[1, 1], &[1, -1])
            .unwrap_or_else(|error| panic!("{error}"));
        run.count_oracle_a();
    }

    for metric in SCORE_METRICS {
        for run in matching_combinations(combinations, TestElement::F32, metric) {
            assert_exact_f32(run.path, metric, &[], &[]).unwrap_or_else(|error| panic!("{error}"));
            run.count_oracle_a();
        }
        for run in matching_combinations(combinations, TestElement::F16, metric) {
            assert_exact_f16(run.path, metric, &[], &[]).unwrap_or_else(|error| panic!("{error}"));
            run.count_oracle_a();
        }
        for run in matching_combinations(combinations, TestElement::I8, metric) {
            assert_exact_i8(run.path, metric, &[], &[]).unwrap_or_else(|error| panic!("{error}"));
            run.count_oracle_a();
        }
    }
}

fn run_near_zero_exact_cases(combinations: &[CombinationRun]) {
    // 17 elements puts the small terms inside the widest SIMD main loop
    // (AVX-512 processes 16 f32 lanes) plus a remainder lane, so these cases
    // exercise vector arithmetic on small magnitudes, not just scalar tails.
    const NEAR_ZERO_DIM: usize = 17;
    const SMALL: f32 = 0.000_976_562_5; // 2^-10, exactly representable in f16

    let f32_l2_a = [0.0f32; NEAR_ZERO_DIM];
    let mut f32_l2_b = [SMALL; NEAR_ZERO_DIM];
    f32_l2_b[NEAR_ZERO_DIM - 1] = 0.0;
    let f16_l2_a = [F16::from_f32(0.0); NEAR_ZERO_DIM];
    let f16_l2_b: Vec<F16> = f32_l2_b.iter().copied().map(F16::from_f32).collect();
    let f32_cancel_a = [1.0f32; NEAR_ZERO_DIM];
    let mut f32_cancel_b = [0.0f32; NEAR_ZERO_DIM];
    for (index, value) in f32_cancel_b.iter_mut().enumerate() {
        // Alternating +/- small terms cancel exactly across SIMD lanes.
        *value = if index % 2 == 0 { SMALL } else { -SMALL };
    }
    f32_cancel_b[NEAR_ZERO_DIM - 1] = 0.0;
    let f16_cancel_a = [F16::from_f32(1.0); NEAR_ZERO_DIM];
    let f16_cancel_b: Vec<F16> = f32_cancel_b.iter().copied().map(F16::from_f32).collect();

    for run in matching_combinations(combinations, TestElement::F32, MetricType::L2) {
        assert_exact_f32(run.path, run.metric, &f32_l2_a, &f32_l2_b)
            .unwrap_or_else(|error| panic!("{error}"));
        run.count_oracle_a();
    }
    for run in matching_combinations(combinations, TestElement::F16, MetricType::L2) {
        assert_exact_f16(run.path, run.metric, &f16_l2_a, &f16_l2_b)
            .unwrap_or_else(|error| panic!("{error}"));
        run.count_oracle_a();
    }
    for run in matching_combinations(combinations, TestElement::F32, MetricType::InnerProduct) {
        assert_exact_f32(run.path, run.metric, &f32_cancel_a, &f32_cancel_b)
            .unwrap_or_else(|error| panic!("{error}"));
        run.count_oracle_a();
    }
    for run in matching_combinations(combinations, TestElement::F16, MetricType::InnerProduct) {
        assert_exact_f16(run.path, run.metric, &f16_cancel_a, &f16_cancel_b)
            .unwrap_or_else(|error| panic!("{error}"));
        run.count_oracle_a();
    }
}

fn assert_batch_equivalence<T: Element>(run: &CombinationRun, query: &[T], targets: &[Vec<T>]) {
    let kernel = ScoreKernel::<T>::with_path(run.metric, run.path).expect("constructible path");
    let target_slices: Vec<&[T]> = targets.iter().map(Vec::as_slice).collect();
    let mut contiguous = Vec::with_capacity(query.len() * targets.len());
    for target in targets {
        contiguous.extend_from_slice(target);
    }
    let expected: Vec<f32> = target_slices
        .iter()
        .map(|target| kernel.score(query, target))
        .collect();
    let mut many = vec![f32::NAN; targets.len()];
    let mut contiguous_out = vec![f32::NAN; targets.len()];
    kernel.score_many(query, &target_slices, &mut many);
    kernel.score_contiguous(query, &contiguous, &mut contiguous_out);

    for (index, ((expected, many), contiguous)) in
        expected.iter().zip(&many).zip(&contiguous_out).enumerate()
    {
        assert_eq!(
            many.to_bits(),
            expected.to_bits(),
            "score_many mismatch: path={:?} element={} metric={:?} dimension={} target={index}",
            run.path,
            run.element.name(),
            run.metric,
            query.len()
        );
        assert_eq!(
            contiguous.to_bits(),
            expected.to_bits(),
            "score_contiguous mismatch: path={:?} element={} metric={:?} dimension={} target={index}",
            run.path,
            run.element.name(),
            run.metric,
            query.len()
        );
    }
}

fn run_batch_equivalence(combinations: &[CombinationRun]) {
    for dimension in BATCH_DIMENSIONS {
        let query_grid = fixed_grid(dimension, 17, 5);
        let target_grids = [
            fixed_grid(dimension, 29, 11),
            fixed_grid(dimension, 13, 7),
            fixed_grid(dimension, 31, 3),
        ];
        let query_f32: Vec<f32> = query_grid.iter().copied().map(f32::from).collect();
        let targets_f32: Vec<Vec<f32>> = target_grids
            .iter()
            .map(|target| target.iter().copied().map(f32::from).collect())
            .collect();
        let query_f16: Vec<F16> = query_f32.iter().copied().map(F16::from_f32).collect();
        let targets_f16: Vec<Vec<F16>> = targets_f32
            .iter()
            .map(|target| target.iter().copied().map(F16::from_f32).collect())
            .collect();
        let query_i8 = fixed_i8(dimension, 73, 19);
        let targets_i8 = vec![
            fixed_i8(dimension, 151, 47),
            fixed_i8(dimension, 109, 23),
            fixed_i8(dimension, 193, 61),
        ];

        for run in combinations {
            match run.element {
                TestElement::F32 => assert_batch_equivalence(run, &query_f32, &targets_f32),
                TestElement::F16 => assert_batch_equivalence(run, &query_f16, &targets_f16),
                TestElement::I8 => assert_batch_equivalence(run, &query_i8, &targets_i8),
            }
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

fn run_i8_conversion_boundaries(combinations: &[CombinationRun]) {
    for score in [(1_u32 << 24) - 1, 1_u32 << 24, (1_u32 << 24) + 1] {
        let (l2_a, l2_b) = i8_l2_pair(score);
        let (dot_a, dot_b) = i8_neg_dot_pair(score);
        for run in matching_combinations(combinations, TestElement::I8, MetricType::L2) {
            assert_exact_i8(run.path, MetricType::L2, &l2_a, &l2_b)
                .unwrap_or_else(|error| panic!("{error}"));
            run.count_oracle_a();
        }
        for run in matching_combinations(combinations, TestElement::I8, MetricType::InnerProduct) {
            assert_exact_i8(run.path, MetricType::InnerProduct, &dot_a, &dot_b)
                .unwrap_or_else(|error| panic!("{error}"));
            run.count_oracle_a();
        }
    }
}

// The cast pins the kernel's one final exact-integer-to-f32 conversion.
#[expect(clippy::cast_possible_truncation)]
fn run_i8_max_dimension(combinations: &[CombinationRun]) {
    let query = vec![i8::MIN; MAX_I8_DIMENSION];
    let target = vec![i8::MAX; MAX_I8_DIMENSION];
    let expected_l2 = f64_oracle(MetricType::L2, &query, &target, f64::from).0 as f32;
    let expected_dot = f64_oracle(MetricType::InnerProduct, &query, &target, f64::from).0 as f32;
    assert_eq!(expected_l2.to_bits(), 2_130_739_200.0f32.to_bits());
    assert_eq!(
        expected_dot.to_bits(),
        (-(32_768f64 * (-128f64 * 127f64)) as f32).to_bits()
    );

    for run in matching_combinations(combinations, TestElement::I8, MetricType::L2) {
        assert_exact_i8(run.path, run.metric, &query, &target)
            .unwrap_or_else(|error| panic!("{error}"));
        run.count_oracle_a();
    }
    for run in matching_combinations(combinations, TestElement::I8, MetricType::InnerProduct) {
        assert_exact_i8(run.path, run.metric, &query, &target)
            .unwrap_or_else(|error| panic!("{error}"));
        run.count_oracle_a();
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
fn vnni_query_bias_identity_matches_signed_dot_product() {
    const EDGE_VALUES: [i8; 7] = [-128, -127, -1, 0, 1, 126, 127];

    for query_value in EDGE_VALUES {
        for target_value in EDGE_VALUES {
            let query = [query_value; DPBUSD_BYTE_LANES];
            let target = [target_value; DPBUSD_BYTE_LANES];
            let (dot, corrected_dpbusd) = vnni_bias_identity_sides(&query, &target);
            assert_eq!(
                dot, corrected_dpbusd,
                "query value {query_value}, target value {target_value}"
            );
        }
    }

    let asymmetric_vectors = (
        prop::collection::vec(any::<i8>(), DPBUSD_BYTE_LANES),
        prop::collection::vec(any::<i8>(), DPBUSD_BYTE_LANES),
    )
        .prop_filter("query and target vectors must differ", |(query, target)| {
            query != target
        });
    TestRunner::new(ProptestConfig {
        cases: 32,
        failure_persistence: None,
        ..ProptestConfig::default()
    })
    .run(&asymmetric_vectors, |(query, target)| {
        let (dot, corrected_dpbusd) = vnni_bias_identity_sides(&query, &target);
        prop_assert_eq!(dot, corrected_dpbusd);
        Ok(())
    })
    .unwrap();
}

#[test]
fn f32_paths_preserve_precision_beyond_f16() {
    const HIGH: f32 = 1.0 + 839.0 * f32::EPSILON;
    const PAIRS: usize = 16;

    let mut query = Vec::with_capacity(PAIRS * 2);
    let mut target = Vec::with_capacity(PAIRS * 2);
    for _ in 0..PAIRS {
        query.extend_from_slice(&[HIGH, 1.0]);
        target.extend_from_slice(&[1.0, -1.0]);
    }

    let expected = f64_oracle(MetricType::InnerProduct, &query, &target, f64::from).0;
    let f16_rounded_query: Vec<f32> = query
        .iter()
        .copied()
        .map(|value| F16::from_f32(value).to_f32())
        .collect();
    let f16_rounded_expected = f64_oracle(
        MetricType::InnerProduct,
        &f16_rounded_query,
        &target,
        f64::from,
    )
    .0;
    let f16_relative_shift = (f16_rounded_expected - expected).abs() / expected.abs();
    assert!(
        f16_relative_shift.is_finite() && f16_relative_shift > 1e-3,
        "precision canary does not distinguish f32 from f16: expected={expected:?}, \
         f16-rounded={f16_rounded_expected:?}, relative-shift={f16_relative_shift:?}"
    );

    // Catches a kernel that internally degrades precision, which the integer-grid oracle cannot see.
    let mut constructible_paths = 0;
    for path in ALL_PATHS {
        let Ok(kernel) = ScoreKernel::<f32>::with_path(MetricType::InnerProduct, path) else {
            continue;
        };
        constructible_paths += 1;
        let got = f64::from(kernel.score(&query, &target));
        let relative_error = (got - expected).abs() / expected.abs();
        assert!(
            relative_error.is_finite() && relative_error <= 1e-5,
            "f32 precision canary failed: path={path:?}, got={got:?}, \
             f64-oracle={expected:?}, relative-error={relative_error:?}"
        );
    }
    assert!(constructible_paths > 0);
}

#[test]
fn all_constructible_paths_match_dual_oracles() {
    let combinations = constructible_combinations();
    assert!(
        combinations
            .iter()
            .filter(|run| run.path == KernelPath::Scalar)
            .count()
            == SCORE_METRICS.len() * 3,
        "every scalar (element, metric) combination must be constructible"
    );

    run_fixed_dimensions(&combinations);
    run_zero_and_cancellation_cases(&combinations);
    run_near_zero_exact_cases(&combinations);
    run_batch_equivalence(&combinations);
    run_i8_conversion_boundaries(&combinations);
    run_i8_max_dimension(&combinations);

    property_runner()
        .run(&oracle_a_strategy(), |(grid_a, grid_b, i8_a, i8_b)| {
            run_oracle_a_case(&combinations, &grid_a, &grid_b, &i8_a, &i8_b)
                .map_err(TestCaseError::fail)
        })
        .unwrap();

    property_runner()
        .run(&oracle_b_strategy(), |(f32_a, f32_b, i8_a, i8_b)| {
            run_oracle_b_case(&combinations, &f32_a, &f32_b, &i8_a, &i8_b)
                .map_err(TestCaseError::fail)
        })
        .unwrap();

    for run in &combinations {
        assert!(run.oracle_a_cases.get() > 0);
        assert!(run.oracle_b_cases.get() > 0);
    }
    for path in ALL_PATHS {
        let path_runs: Vec<&CombinationRun> =
            combinations.iter().filter(|run| run.path == path).collect();
        let exact_case_count: usize = path_runs.iter().map(|run| run.oracle_a_cases.get()).sum();
        let oracle_b_case_count: usize = path_runs.iter().map(|run| run.oracle_b_cases.get()).sum();
        println!(
            "path summary: {path:?}; exercised combinations: {}; skipped combinations: {}; \
             oracle A cases: {exact_case_count}; oracle B cases: {oracle_b_case_count}",
            path_runs.len(),
            SCORE_METRICS.len() * 3 - path_runs.len(),
        );
    }
}
