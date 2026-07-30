use std::collections::HashSet;
use std::hint::black_box;
use std::time::{Duration, Instant};

use vectordb_simd::{Element, F16, KernelPath, MetricType, ScoreKernel, baseline};

const DIMENSIONS: [usize; 3] = [128, 768, 1536];
const PATHS: [KernelPath; 4] = [
    KernelPath::Scalar,
    KernelPath::Avx2,
    KernelPath::Avx512,
    KernelPath::Neon,
];
const SCORE_METRICS: [MetricType; 2] = [MetricType::L2, MetricType::InnerProduct];
const WARM_UP_TIME: Duration = Duration::from_millis(10);
const MEASUREMENT_TIME: Duration = Duration::from_millis(200);
const TIMING_BATCH_SIZE: u32 = 256;

type BaselineFn<T> = fn(&[T], &[T]) -> f32;

fn path_name(path: KernelPath) -> &'static str {
    match path {
        KernelPath::Scalar => "scalar",
        KernelPath::Avx2 => "avx2",
        KernelPath::Avx512 => "avx512",
        KernelPath::Neon => "neon",
        _ => panic!("unknown kernel path: {path:?}"),
    }
}

fn path_index(path: KernelPath) -> usize {
    PATHS
        .iter()
        .position(|candidate| *candidate == path)
        .expect("known kernel path")
}

// Entries may carry a `path:element` qualifier (enforced per combination by
// the equivalence suite); this report gates on the path part only.
fn required_paths() -> HashSet<KernelPath> {
    let value = std::env::var("VECTORDB_SIMD_REQUIRE").unwrap_or_default();
    value
        .split(',')
        .filter_map(|entry| {
            let entry = entry.trim().to_ascii_lowercase();
            if entry.is_empty() {
                return None;
            }
            let path = match entry.split_once(':') {
                Some((path, element)) => {
                    assert!(
                        matches!(element, "f32" | "f16" | "i8"),
                        "unknown VECTORDB_SIMD_REQUIRE element: {element}"
                    );
                    path
                }
                None => entry.as_str(),
            };
            Some(match path {
                "scalar" => KernelPath::Scalar,
                "avx2" => KernelPath::Avx2,
                "avx512" => KernelPath::Avx512,
                "neon" => KernelPath::Neon,
                _ => panic!("unknown VECTORDB_SIMD_REQUIRE path: {path}"),
            })
        })
        .collect()
}

fn metric_name(metric: MetricType) -> &'static str {
    match metric {
        MetricType::L2 => "L2",
        MetricType::InnerProduct => "InnerProduct",
        MetricType::Cosine => "Cosine",
        _ => panic!("unknown metric variant: {metric:?}"),
    }
}

fn baseline_for<T>(
    metric: MetricType,
    squared_l2: BaselineFn<T>,
    neg_dot: BaselineFn<T>,
) -> BaselineFn<T> {
    match metric {
        MetricType::L2 => squared_l2,
        MetricType::InnerProduct | MetricType::Cosine => neg_dot,
        _ => panic!("unknown metric variant: {metric:?}"),
    }
}

fn values_f32(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|index| {
            let value = (index.wrapping_mul(17).wrapping_add(seed)) % 33;
            #[expect(clippy::cast_precision_loss)]
            let value = value as f32;
            value - 16.0
        })
        .collect()
}

fn values_f16(len: usize, seed: usize) -> Vec<F16> {
    values_f32(len, seed)
        .into_iter()
        .map(F16::from_f32)
        .collect()
}

fn values_i8(len: usize, seed: usize) -> Vec<i8> {
    (0..len)
        .map(|index| {
            let value = (index.wrapping_mul(17).wrapping_add(seed)) % 33;
            i8::try_from(value).expect("generated values fit in i8") - 16
        })
        .collect()
}

fn average_nanoseconds(mut score: impl FnMut() -> f32) -> f64 {
    let warm_up_start = Instant::now();
    while warm_up_start.elapsed() < WARM_UP_TIME {
        for _ in 0..TIMING_BATCH_SIZE {
            black_box(score());
        }
    }

    let start = Instant::now();
    let mut measurement_batches = 0u32;
    // Clock reads per call compressed small-dimension speedup factors, so time
    // is sampled only after an inner batch of kernel calls.
    while start.elapsed() < MEASUREMENT_TIME {
        for _ in 0..TIMING_BATCH_SIZE {
            black_box(score());
        }
        measurement_batches += 1;
    }
    let total_calls = f64::from(measurement_batches) * f64::from(TIMING_BATCH_SIZE);
    start.elapsed().as_secs_f64() * 1_000_000_000.0 / total_calls
}

fn report_element<T: Element>(
    element: &str,
    make_values: fn(usize, usize) -> Vec<T>,
    naive_squared_l2: BaselineFn<T>,
    naive_neg_dot: BaselineFn<T>,
    safe_squared_l2: BaselineFn<T>,
    safe_neg_dot: BaselineFn<T>,
    recorded_lines: &mut [usize; 4],
) {
    for metric in SCORE_METRICS {
        let naive_fn = baseline_for(metric, naive_squared_l2, naive_neg_dot);
        let safe_fn = baseline_for(metric, safe_squared_l2, safe_neg_dot);
        for path in PATHS {
            let Ok(kernel) = ScoreKernel::<T>::with_path(metric, path) else {
                continue;
            };
            for dimension in DIMENSIONS {
                let query = make_values(dimension, 3);
                let target = make_values(dimension, 11);
                let naive_ns = average_nanoseconds(|| {
                    naive_fn(black_box(query.as_slice()), black_box(target.as_slice()))
                });
                let safe_ns = average_nanoseconds(|| {
                    safe_fn(black_box(query.as_slice()), black_box(target.as_slice()))
                });
                let path_ns = average_nanoseconds(|| {
                    kernel.score(black_box(query.as_slice()), black_box(target.as_slice()))
                });
                let naive_factor = naive_ns / path_ns;
                let safe_factor = safe_ns / path_ns;
                println!(
                    "element={element} metric={} path={} dim={dimension} \
                     naive_factor={naive_factor:.3} safe_factor={safe_factor:.3}",
                    metric_name(metric),
                    path_name(path),
                );
                recorded_lines[path_index(path)] += 1;
            }
        }
    }
}

#[test]
#[ignore = "timing report is run explicitly in release mode"]
fn speedup_report() {
    let mut recorded_lines = [0usize; 4];
    report_element(
        "f32",
        values_f32,
        baseline::naive_squared_l2_f32,
        baseline::naive_neg_dot_f32,
        baseline::safe_squared_l2_f32,
        baseline::safe_neg_dot_f32,
        &mut recorded_lines,
    );
    report_element(
        "f16",
        values_f16,
        baseline::naive_squared_l2_f16,
        baseline::naive_neg_dot_f16,
        baseline::safe_squared_l2_f16,
        baseline::safe_neg_dot_f16,
        &mut recorded_lines,
    );
    report_element(
        "i8",
        values_i8,
        baseline::naive_squared_l2_i8,
        baseline::naive_neg_dot_i8,
        baseline::safe_squared_l2_i8,
        baseline::safe_neg_dot_i8,
        &mut recorded_lines,
    );

    for required in required_paths() {
        assert!(
            recorded_lines[path_index(required)] > 0,
            "required path {} produced zero speedup report lines",
            path_name(required)
        );
    }
}
