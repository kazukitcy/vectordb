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
const METRICS: [MetricType; 3] = [MetricType::L2, MetricType::InnerProduct, MetricType::Cosine];
const WARM_UP_TIME: Duration = Duration::from_millis(10);
const MEASUREMENT_TIME: Duration = Duration::from_millis(200);

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
            #[allow(clippy::cast_precision_loss)]
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
        black_box(score());
    }

    let start = Instant::now();
    let mut iterations = 0u32;
    while start.elapsed() < MEASUREMENT_TIME {
        black_box(score());
        iterations += 1;
    }
    start.elapsed().as_secs_f64() * 1_000_000_000.0 / f64::from(iterations)
}

fn report_element<T: Element>(
    element: &str,
    make_values: fn(usize, usize) -> Vec<T>,
    squared_l2: BaselineFn<T>,
    neg_dot: BaselineFn<T>,
) {
    for metric in METRICS {
        let baseline_fn = baseline_for(metric, squared_l2, neg_dot);
        for path in PATHS {
            let Ok(kernel) = ScoreKernel::<T>::with_path(metric, path) else {
                continue;
            };
            for dimension in DIMENSIONS {
                let query = make_values(dimension, 3);
                let target = make_values(dimension, 11);
                let baseline_ns = average_nanoseconds(|| {
                    baseline_fn(black_box(query.as_slice()), black_box(target.as_slice()))
                });
                let path_ns = average_nanoseconds(|| {
                    kernel.score(black_box(query.as_slice()), black_box(target.as_slice()))
                });
                let factor = baseline_ns / path_ns;
                println!(
                    "element={element} metric={} path={} dim={dimension} factor={factor:.3}",
                    metric_name(metric),
                    path_name(path),
                );
            }
        }
    }
}

#[test]
#[ignore = "timing report is run explicitly in release mode"]
fn speedup_report() {
    report_element(
        "f32",
        values_f32,
        baseline::squared_l2_f32,
        baseline::neg_dot_f32,
    );
    report_element(
        "f16",
        values_f16,
        baseline::squared_l2_f16,
        baseline::neg_dot_f16,
    );
    report_element(
        "i8",
        values_i8,
        baseline::squared_l2_i8,
        baseline::neg_dot_i8,
    );
}
