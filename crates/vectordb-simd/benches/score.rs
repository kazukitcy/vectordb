use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use vectordb_simd::{Element, F16, KernelPath, MetricType, ScoreKernel, baseline};

const DIMENSIONS: [usize; 3] = [128, 768, 1536];
const BATCH_SIZE: usize = 1000;
const PATHS: [KernelPath; 4] = [
    KernelPath::Scalar,
    KernelPath::Avx2,
    KernelPath::Avx512,
    KernelPath::Neon,
];
const SCORE_METRICS: [MetricType; 2] = [MetricType::L2, MetricType::InnerProduct];

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

fn score_baseline_contiguous<T>(
    query: &[T],
    vectors: &[T],
    out: &mut [f32],
    baseline_fn: BaselineFn<T>,
) {
    let dimension = query.len();
    for (row, value) in out.iter_mut().enumerate() {
        let start = row * dimension;
        *value = baseline_fn(query, &vectors[start..start + dimension]);
    }
}

fn bench_single<T: Element>(
    criterion: &mut Criterion,
    element: &str,
    make_values: fn(usize, usize) -> Vec<T>,
    squared_l2: BaselineFn<T>,
    neg_dot: BaselineFn<T>,
) {
    for metric in SCORE_METRICS {
        let mut group =
            criterion.benchmark_group(format!("{element}/{}/single", metric_name(metric)));
        let baseline_fn = baseline_for(metric, squared_l2, neg_dot);

        for dimension in DIMENSIONS {
            let query = make_values(dimension, 3);
            let target = make_values(dimension, 11);
            group.throughput(Throughput::Elements(dimension as u64));
            group.bench_function(BenchmarkId::new("baseline", dimension), |bencher| {
                bencher.iter(|| {
                    black_box(baseline_fn(
                        black_box(query.as_slice()),
                        black_box(target.as_slice()),
                    ))
                });
            });

            for path in PATHS {
                let Ok(kernel) = ScoreKernel::<T>::with_path(metric, path) else {
                    continue;
                };
                group.bench_function(BenchmarkId::new(path_name(path), dimension), |bencher| {
                    bencher.iter(|| {
                        black_box(
                            kernel.score(black_box(query.as_slice()), black_box(target.as_slice())),
                        )
                    });
                });
            }
        }
        group.finish();
    }
}

fn bench_batch<T: Element>(
    criterion: &mut Criterion,
    element: &str,
    make_values: fn(usize, usize) -> Vec<T>,
    squared_l2: BaselineFn<T>,
    neg_dot: BaselineFn<T>,
) {
    for metric in SCORE_METRICS {
        let mut group =
            criterion.benchmark_group(format!("{element}/{}/contiguous", metric_name(metric)));
        let baseline_fn = baseline_for(metric, squared_l2, neg_dot);

        for dimension in DIMENSIONS {
            let query = make_values(dimension, 3);
            let vectors = make_values(dimension * BATCH_SIZE, 11);
            group.throughput(Throughput::Elements((dimension * BATCH_SIZE) as u64));
            group.bench_function(BenchmarkId::new("baseline", dimension), |bencher| {
                let mut out = vec![0.0; BATCH_SIZE];
                bencher.iter(|| {
                    score_baseline_contiguous(
                        black_box(query.as_slice()),
                        black_box(vectors.as_slice()),
                        black_box(out.as_mut_slice()),
                        baseline_fn,
                    );
                });
            });

            for path in PATHS {
                let Ok(kernel) = ScoreKernel::<T>::with_path(metric, path) else {
                    continue;
                };
                group.bench_function(BenchmarkId::new(path_name(path), dimension), |bencher| {
                    let mut out = vec![0.0; BATCH_SIZE];
                    bencher.iter(|| {
                        kernel.score_contiguous(
                            black_box(query.as_slice()),
                            black_box(vectors.as_slice()),
                            black_box(out.as_mut_slice()),
                        );
                    });
                });
            }
        }
        group.finish();

        // score_many is the graph-search hot path and carries the bounded
        // automatic prefetch; record it separately from the contiguous scan.
        let mut group =
            criterion.benchmark_group(format!("{element}/{}/many", metric_name(metric)));
        for dimension in DIMENSIONS {
            let query = make_values(dimension, 3);
            let targets: Vec<Vec<T>> = (0..BATCH_SIZE)
                .map(|seed| make_values(dimension, 11 + seed))
                .collect();
            let target_slices: Vec<&[T]> = targets.iter().map(Vec::as_slice).collect();
            group.throughput(Throughput::Elements((dimension * BATCH_SIZE) as u64));

            for path in PATHS {
                let Ok(kernel) = ScoreKernel::<T>::with_path(metric, path) else {
                    continue;
                };
                group.bench_function(BenchmarkId::new(path_name(path), dimension), |bencher| {
                    let mut out = vec![0.0; BATCH_SIZE];
                    bencher.iter(|| {
                        kernel.score_many(
                            black_box(query.as_slice()),
                            black_box(target_slices.as_slice()),
                            black_box(out.as_mut_slice()),
                        );
                    });
                });
            }
        }
        group.finish();
    }
}

fn score_kernels(criterion: &mut Criterion) {
    bench_single(
        criterion,
        "f32",
        values_f32,
        baseline::squared_l2_f32,
        baseline::neg_dot_f32,
    );
    bench_batch(
        criterion,
        "f32",
        values_f32,
        baseline::squared_l2_f32,
        baseline::neg_dot_f32,
    );
    bench_single(
        criterion,
        "f16",
        values_f16,
        baseline::squared_l2_f16,
        baseline::neg_dot_f16,
    );
    bench_batch(
        criterion,
        "f16",
        values_f16,
        baseline::squared_l2_f16,
        baseline::neg_dot_f16,
    );
    bench_single(
        criterion,
        "i8",
        values_i8,
        baseline::squared_l2_i8,
        baseline::neg_dot_i8,
    );
    bench_batch(
        criterion,
        "i8",
        values_i8,
        baseline::squared_l2_i8,
        baseline::neg_dot_i8,
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(100))
        .measurement_time(Duration::from_millis(200))
        .sample_size(10);
    targets = score_kernels
}
criterion_main!(benches);
