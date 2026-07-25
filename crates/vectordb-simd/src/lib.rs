//! Scalar reference and runtime-dispatched SIMD score kernels.
//!
//! Every kernel returns an `f32` score where smaller is better:
//!
//! | Metric | Score |
//! | --- | --- |
//! | [`MetricType::L2`] | Squared Euclidean distance |
//! | [`MetricType::InnerProduct`] | Negated dot product |
//! | [`MetricType::Cosine`] | Negated dot product over pre-normalized inputs |
//!
//! Floating-point kernels accumulate in `f32`. Their scores may therefore be
//! non-finite when intermediate sums exceed the `f32` range, even if every
//! input component is finite.

mod scalar;

use core::fmt;

pub use vectordb_core::{Error, F16, MetricType, Result};

/// Maximum dimension accepted by `i8` kernels.
///
/// The `i32` accumulation is exact up to this dimension.
pub const MAX_I8_DIMENSION: usize = 32_768;

/// A vector element type supported by the score kernels.
///
/// This trait is sealed and implemented for `f32`, [`F16`], and `i8` only.
pub trait Element: sealed::Sealed + Copy + Send + Sync + 'static {}

/// An instruction-set path a kernel can execute on.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum KernelPath {
    /// Portable scalar fallback; available on every CPU.
    Scalar,
    /// x86-64 AVX2 family.
    Avx2,
    /// x86-64 AVX-512 family.
    Avx512,
    /// `AArch64` NEON.
    Neon,
}

type KernelFn<T> = unsafe fn(&[T], &[T]) -> f32;

mod sealed {
    use super::{ImplementedPaths, KernelFn, KernelPath, MetricType, scalar};

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub enum ElementKind {
        F32,
        F16,
        I8,
    }

    #[derive(Clone, Copy)]
    pub struct KernelTable<T> {
        scalar: KernelFn<T>,
        avx2: Option<KernelFn<T>>,
        avx512: Option<KernelFn<T>>,
        neon: Option<KernelFn<T>>,
    }

    impl<T> KernelTable<T> {
        const fn scalar(kernel: KernelFn<T>) -> Self {
            Self {
                scalar: kernel,
                avx2: None,
                avx512: None,
                neon: None,
            }
        }

        pub(super) const fn implemented_paths(&self) -> ImplementedPaths {
            ImplementedPaths {
                scalar: true,
                avx2: self.avx2.is_some(),
                avx512: self.avx512.is_some(),
                neon: self.neon.is_some(),
            }
        }

        pub(super) const fn kernel(&self, path: KernelPath) -> Option<KernelFn<T>> {
            match path {
                KernelPath::Scalar => Some(self.scalar),
                KernelPath::Avx2 => self.avx2,
                KernelPath::Avx512 => self.avx512,
                KernelPath::Neon => self.neon,
            }
        }
    }

    fn scalar_table<T>(
        metric: MetricType,
        squared_l2: KernelFn<T>,
        neg_dot: KernelFn<T>,
    ) -> KernelTable<T> {
        match metric {
            MetricType::L2 => KernelTable::scalar(squared_l2),
            MetricType::InnerProduct | MetricType::Cosine => KernelTable::scalar(neg_dot),
            _ => panic!("unknown metric variant: {metric:?}"),
        }
    }

    pub trait Sealed {
        const KIND: ElementKind;

        fn kernel_table(metric: MetricType) -> KernelTable<Self>
        where
            Self: Sized;

        fn validate_dimension(_dimension: usize) {}
    }

    impl Sealed for f32 {
        const KIND: ElementKind = ElementKind::F32;

        fn kernel_table(metric: MetricType) -> KernelTable<Self> {
            scalar_table(metric, scalar::squared_l2_f32, scalar::neg_dot_f32)
        }
    }

    impl Sealed for vectordb_core::F16 {
        const KIND: ElementKind = ElementKind::F16;

        fn kernel_table(metric: MetricType) -> KernelTable<Self> {
            scalar_table(metric, scalar::squared_l2_f16, scalar::neg_dot_f16)
        }
    }

    impl Sealed for i8 {
        const KIND: ElementKind = ElementKind::I8;

        fn kernel_table(metric: MetricType) -> KernelTable<Self> {
            scalar_table(metric, scalar::squared_l2_i8, scalar::neg_dot_i8)
        }

        fn validate_dimension(dimension: usize) {
            assert!(
                dimension <= super::MAX_I8_DIMENSION,
                "i8 vector dimension {dimension} exceeds MAX_I8_DIMENSION"
            );
        }
    }
}

impl Element for f32 {}
impl Element for F16 {}
impl Element for i8 {}

use sealed::ElementKind;

/// A score kernel resolved for one element type, metric, and instruction path.
#[derive(Clone, Copy)]
pub struct ScoreKernel<T: Element> {
    metric: MetricType,
    path: KernelPath,
    kernel: KernelFn<T>,
}

impl<T: Element> ScoreKernel<T> {
    /// Selects the highest-priority path the current CPU supports.
    ///
    /// This never fails for metrics known to this crate version.
    ///
    /// # Panics
    ///
    /// Panics on a [`MetricType`] variant unknown to this crate version.
    pub fn new(metric: MetricType) -> Self {
        match Self::construct(metric, None) {
            Ok(kernel) => kernel,
            Err(error) => panic!("failed to resolve a score kernel: {error}"),
        }
    }

    /// Forces a specific instruction-set path.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Unsupported`] when the current CPU or target
    /// architecture does not support `path` for `T`.
    pub fn with_path(metric: MetricType, path: KernelPath) -> Result<Self> {
        Self::construct(metric, Some(path))
    }

    /// Returns the selected instruction-set path.
    pub const fn path(&self) -> KernelPath {
        self.path
    }

    /// Returns the metric scored by this kernel.
    pub const fn metric(&self) -> MetricType {
        self.metric
    }

    /// Computes the score between two vectors of equal dimension.
    ///
    /// Floating-point inputs may produce a non-finite score when intermediate
    /// sums exceed the `f32` range.
    ///
    /// # Panics
    ///
    /// Panics if the vector lengths differ, or if an `i8` vector exceeds
    /// [`MAX_I8_DIMENSION`]. Zero-dimension inputs yield `0.0` for squared L2
    /// and `-0.0` for negated dot.
    pub fn score(&self, a: &[T], b: &[T]) -> f32 {
        assert!(
            a.len() == b.len(),
            "score vector dimensions differ: {} and {}",
            a.len(),
            b.len()
        );
        T::validate_dimension(a.len());
        self.invoke(a, b)
    }

    /// Scores `query` against each target slice.
    ///
    /// The entire batch is validated before any output is written.
    ///
    /// # Panics
    ///
    /// Panics if any target length differs from `query.len()`, if
    /// `targets.len() != out.len()`, or if an `i8` vector exceeds
    /// [`MAX_I8_DIMENSION`].
    pub fn score_many(&self, query: &[T], targets: &[&[T]], out: &mut [f32]) {
        assert!(
            targets.len() == out.len(),
            "target and output counts differ: {} and {}",
            targets.len(),
            out.len()
        );
        T::validate_dimension(query.len());
        for target in targets {
            assert!(
                target.len() == query.len(),
                "query and target dimensions differ: {} and {}",
                query.len(),
                target.len()
            );
        }

        for (index, target) in targets.iter().enumerate() {
            if index + 1 < targets.len() {
                self.prefetch(targets[index + 1]);
            }
            out[index] = self.invoke(query, target);
        }
    }

    /// Scores `query` against row-major contiguous vectors.
    ///
    /// The number of rows is `out.len()`.
    ///
    /// # Panics
    ///
    /// Panics if `query` is empty, if `vectors.len()` does not equal
    /// `query.len().checked_mul(out.len())`, if that multiplication overflows,
    /// or if an `i8` vector exceeds [`MAX_I8_DIMENSION`].
    pub fn score_contiguous(&self, query: &[T], vectors: &[T], out: &mut [f32]) {
        assert!(
            !query.is_empty(),
            "contiguous scoring requires a nonempty query"
        );
        T::validate_dimension(query.len());
        let expected_len = query
            .len()
            .checked_mul(out.len())
            .expect("contiguous vector length overflow");
        assert!(
            vectors.len() == expected_len,
            "contiguous vector length is {}, expected {expected_len}",
            vectors.len()
        );

        let dimension = query.len();
        for index in 0..out.len() {
            if index + 1 < out.len() {
                let next_start = (index + 1) * dimension;
                self.prefetch(&vectors[next_start..next_start + dimension]);
            }
            let start = index * dimension;
            out[index] = self.invoke(query, &vectors[start..start + dimension]);
        }
    }

    /// Hints that `target` will be scored soon.
    ///
    /// This is a no-op on paths or platforms without a prefetch primitive.
    pub fn prefetch(&self, _target: &[T]) {}

    fn construct(metric: MetricType, requested: Option<KernelPath>) -> Result<Self> {
        let table = T::kernel_table(metric);
        let implemented = table.implemented_paths();
        let path = resolve_path(
            current_arch(),
            &detected_features(),
            &implemented,
            T::KIND,
            requested,
        )?;
        let Some(kernel) = table.kernel(path) else {
            return Err(Error::internal(
                "resolved score-kernel path has no kernel function",
            ));
        };
        Ok(Self {
            metric,
            path,
            kernel,
        })
    }

    fn invoke(&self, a: &[T], b: &[T]) -> f32 {
        // SAFETY: construction pairs this pointer with a resolver-verified path;
        // public chokepoints validate kernel length and dimension preconditions.
        // See ADR 0002.
        unsafe { (self.kernel)(a, b) }
    }
}

impl<T: Element> fmt::Debug for ScoreKernel<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScoreKernel")
            .field("metric", &self.metric)
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

/// Returns the L2 norm of an `f32` vector as `f64`.
///
/// The norm is computed in `f64` from the `f32` components and can therefore
/// exceed the `f32` range.
pub fn l2_norm(vector: &[f32]) -> f64 {
    vector
        .iter()
        .map(|&value| {
            let value = f64::from(value);
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

/// Normalizes an `f32` vector to unit L2 norm in place.
///
/// Returns the original norm as `f64`. A zero-norm vector is left unchanged
/// and returns `0.0`. Division uses the `f64` norm for each element; the stored
/// normalized values remain subject to `f32` rounding.
#[allow(clippy::cast_possible_truncation)]
pub fn normalize_l2(vector: &mut [f32]) -> f64 {
    let norm = l2_norm(vector);
    if norm == 0.0 {
        return norm;
    }
    for value in vector {
        *value = (f64::from(*value) / norm) as f32;
    }
    norm
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Arch {
    X86_64,
    Aarch64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FeatureSet {
    avx2: bool,
    fma: bool,
    f16c: bool,
    avx512f: bool,
    avx512bw: bool,
    avx512vnni: bool,
    neon: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ImplementedPaths {
    scalar: bool,
    avx2: bool,
    avx512: bool,
    neon: bool,
}

// References preserve D5's resolver contract as feature tables grow. See ADR 0002.
#[allow(clippy::trivially_copy_pass_by_ref)]
pub(crate) fn resolve_path(
    arch: Arch,
    features: &FeatureSet,
    implemented: &ImplementedPaths,
    element: ElementKind,
    requested: Option<KernelPath>,
) -> Result<KernelPath> {
    let supported = |path| match path {
        KernelPath::Scalar => implemented.scalar,
        KernelPath::Avx2 => {
            arch == Arch::X86_64
                && implemented.avx2
                && features.avx2
                && match element {
                    ElementKind::F32 => features.fma,
                    ElementKind::F16 => features.fma && features.f16c,
                    ElementKind::I8 => true,
                }
        }
        KernelPath::Avx512 => {
            arch == Arch::X86_64
                && implemented.avx512
                && features.avx512f
                && match element {
                    ElementKind::F32 => true,
                    ElementKind::F16 => features.f16c,
                    ElementKind::I8 => features.avx512bw && features.avx512vnni,
                }
        }
        KernelPath::Neon => arch == Arch::Aarch64 && implemented.neon && features.neon,
    };

    if let Some(path) = requested {
        return if supported(path) {
            Ok(path)
        } else {
            Err(Error::unsupported(format!(
                "{path:?} is unavailable for {element:?} on {arch:?}"
            )))
        };
    }

    let candidates: &[KernelPath] = match arch {
        Arch::X86_64 => &[KernelPath::Avx512, KernelPath::Avx2, KernelPath::Scalar],
        Arch::Aarch64 => &[KernelPath::Neon, KernelPath::Scalar],
    };
    candidates
        .iter()
        .copied()
        .find(|&path| supported(path))
        .ok_or_else(|| Error::unsupported("no implemented score-kernel path is available"))
}

#[cfg(target_arch = "x86_64")]
const fn current_arch() -> Arch {
    Arch::X86_64
}

#[cfg(target_arch = "aarch64")]
const fn current_arch() -> Arch {
    Arch::Aarch64
}

#[cfg(target_arch = "x86_64")]
fn detected_features() -> FeatureSet {
    FeatureSet {
        avx2: std::arch::is_x86_feature_detected!("avx2"),
        fma: std::arch::is_x86_feature_detected!("fma"),
        f16c: std::arch::is_x86_feature_detected!("f16c"),
        avx512f: std::arch::is_x86_feature_detected!("avx512f"),
        avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
        avx512vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
        neon: false,
    }
}

#[cfg(target_arch = "aarch64")]
fn detected_features() -> FeatureSet {
    FeatureSet {
        neon: std::arch::is_aarch64_feature_detected!("neon"),
        ..FeatureSet::default()
    }
}

#[cfg(test)]
mod tests {
    use super::{Arch, ElementKind, FeatureSet, ImplementedPaths, KernelPath, resolve_path};
    use vectordb_core::Error;

    fn full_features() -> FeatureSet {
        FeatureSet {
            avx2: true,
            fma: true,
            f16c: true,
            avx512f: true,
            avx512bw: true,
            avx512vnni: true,
            neon: true,
        }
    }

    fn full_paths() -> ImplementedPaths {
        ImplementedPaths {
            scalar: true,
            avx2: true,
            avx512: true,
            neon: true,
        }
    }

    fn scalar_only() -> ImplementedPaths {
        ImplementedPaths {
            scalar: true,
            avx2: false,
            avx512: false,
            neon: false,
        }
    }

    // Owned results keep the resolver assertions direct and readable.
    #[allow(clippy::needless_pass_by_value)]
    fn assert_unsupported(result: vectordb_core::Result<KernelPath>) {
        assert!(matches!(result, Err(Error::Unsupported { .. })));
    }

    #[test]
    fn full_features_auto_select_the_best_path_for_every_element() {
        let features = full_features();
        let implemented = full_paths();

        for element in [ElementKind::F32, ElementKind::F16, ElementKind::I8] {
            assert_eq!(
                resolve_path(Arch::X86_64, &features, &implemented, element, None).unwrap(),
                KernelPath::Avx512
            );
            assert_eq!(
                resolve_path(Arch::Aarch64, &features, &implemented, element, None).unwrap(),
                KernelPath::Neon
            );
        }
    }

    #[test]
    fn avx2_without_fma_is_only_supported_for_i8() {
        let features = FeatureSet {
            avx2: true,
            f16c: true,
            ..FeatureSet::default()
        };
        let implemented = full_paths();

        for element in [ElementKind::F32, ElementKind::F16] {
            assert_unsupported(resolve_path(
                Arch::X86_64,
                &features,
                &implemented,
                element,
                Some(KernelPath::Avx2),
            ));
        }
        assert_eq!(
            resolve_path(
                Arch::X86_64,
                &features,
                &implemented,
                ElementKind::I8,
                Some(KernelPath::Avx2),
            )
            .unwrap(),
            KernelPath::Avx2
        );
    }

    #[test]
    fn avx2_without_f16c_is_unsupported_for_f16() {
        let features = FeatureSet {
            avx2: true,
            fma: true,
            ..FeatureSet::default()
        };
        let implemented = full_paths();

        assert_eq!(
            resolve_path(
                Arch::X86_64,
                &features,
                &implemented,
                ElementKind::F32,
                Some(KernelPath::Avx2),
            )
            .unwrap(),
            KernelPath::Avx2
        );
        assert_unsupported(resolve_path(
            Arch::X86_64,
            &features,
            &implemented,
            ElementKind::F16,
            Some(KernelPath::Avx2),
        ));
        assert_eq!(
            resolve_path(
                Arch::X86_64,
                &features,
                &implemented,
                ElementKind::I8,
                Some(KernelPath::Avx2),
            )
            .unwrap(),
            KernelPath::Avx2
        );
    }

    #[test]
    fn avx512_missing_element_features_falls_back_or_is_unsupported() {
        let without_f16c = FeatureSet {
            f16c: false,
            ..full_features()
        };
        let without_bw = FeatureSet {
            avx512bw: false,
            ..full_features()
        };
        let without_vnni = FeatureSet {
            avx512vnni: false,
            ..full_features()
        };
        let implemented = full_paths();

        assert_eq!(
            resolve_path(
                Arch::X86_64,
                &without_f16c,
                &implemented,
                ElementKind::F32,
                Some(KernelPath::Avx512),
            )
            .unwrap(),
            KernelPath::Avx512
        );
        assert_unsupported(resolve_path(
            Arch::X86_64,
            &without_f16c,
            &implemented,
            ElementKind::F16,
            Some(KernelPath::Avx512),
        ));
        for features in [&without_bw, &without_vnni] {
            assert_unsupported(resolve_path(
                Arch::X86_64,
                features,
                &implemented,
                ElementKind::I8,
                Some(KernelPath::Avx512),
            ));
        }
        assert_eq!(
            resolve_path(
                Arch::X86_64,
                &without_vnni,
                &implemented,
                ElementKind::I8,
                None,
            )
            .unwrap(),
            KernelPath::Avx2
        );
    }

    #[test]
    fn empty_features_fall_back_to_scalar_for_every_element() {
        let features = FeatureSet::default();
        let implemented = full_paths();

        for arch in [Arch::X86_64, Arch::Aarch64] {
            for element in [ElementKind::F32, ElementKind::F16, ElementKind::I8] {
                assert_eq!(
                    resolve_path(arch, &features, &implemented, element, None).unwrap(),
                    KernelPath::Scalar
                );
            }
        }
    }

    #[test]
    fn supported_but_unimplemented_paths_fall_back_or_are_unsupported() {
        let features = full_features();
        let implemented = scalar_only();

        for arch in [Arch::X86_64, Arch::Aarch64] {
            for element in [ElementKind::F32, ElementKind::F16, ElementKind::I8] {
                assert_eq!(
                    resolve_path(arch, &features, &implemented, element, None).unwrap(),
                    KernelPath::Scalar
                );
                for path in [KernelPath::Avx2, KernelPath::Avx512, KernelPath::Neon] {
                    assert_unsupported(resolve_path(
                        arch,
                        &features,
                        &implemented,
                        element,
                        Some(path),
                    ));
                }
            }
        }
    }

    #[test]
    fn foreign_architecture_requests_are_unsupported_for_every_element() {
        let features = full_features();
        let implemented = full_paths();

        for element in [ElementKind::F32, ElementKind::F16, ElementKind::I8] {
            assert_unsupported(resolve_path(
                Arch::X86_64,
                &features,
                &implemented,
                element,
                Some(KernelPath::Neon),
            ));
            for path in [KernelPath::Avx2, KernelPath::Avx512] {
                assert_unsupported(resolve_path(
                    Arch::Aarch64,
                    &features,
                    &implemented,
                    element,
                    Some(path),
                ));
            }
        }
    }
}
