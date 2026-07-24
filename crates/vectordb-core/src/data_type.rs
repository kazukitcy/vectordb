use std::fmt;

// Vector variants with binary or F64 components are deliberately omitted: version 1 does not
// support them, and the enum is non-exhaustive so they can be added later without a breaking
// change.
/// The logical type of a schema field.
///
/// Vectors support `F32`, `F16`, and `I8` dense components and `F32` and `F16` sparse components.
/// Binary and `F64` values are supported as scalars and arrays only.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DataType {
    /// A Boolean scalar.
    Bool,
    /// A signed 32-bit integer scalar.
    I32,
    /// A signed 64-bit integer scalar.
    I64,
    /// An unsigned 32-bit integer scalar.
    U32,
    /// An unsigned 64-bit integer scalar.
    U64,
    /// An IEEE 754 single-precision scalar.
    F32,
    /// An IEEE 754 double-precision scalar.
    F64,
    /// A UTF-8 string scalar.
    String,
    /// An uninterpreted byte string.
    Binary,
    /// An array of Boolean values.
    BoolArray,
    /// An array of signed 32-bit integers.
    I32Array,
    /// An array of signed 64-bit integers.
    I64Array,
    /// An array of unsigned 32-bit integers.
    U32Array,
    /// An array of unsigned 64-bit integers.
    U64Array,
    /// An array of IEEE 754 single-precision values.
    F32Array,
    /// An array of IEEE 754 double-precision values.
    F64Array,
    /// An array of UTF-8 strings.
    StringArray,
    /// An array of uninterpreted byte strings.
    BinaryArray,
    /// A dense vector with single-precision components.
    DenseVectorF32,
    /// A dense vector with half-precision components.
    DenseVectorF16,
    /// A dense vector with signed 8-bit integer components.
    DenseVectorI8,
    /// A sparse vector with explicitly stored single-precision values.
    SparseVectorF32,
    /// A sparse vector with explicitly stored half-precision values.
    SparseVectorF16,
}

impl DataType {
    /// Returns whether this type is any supported vector type.
    pub const fn is_vector(&self) -> bool {
        self.is_dense_vector() || self.is_sparse_vector()
    }

    /// Returns whether this type is a supported dense vector type.
    pub const fn is_dense_vector(&self) -> bool {
        matches!(
            self,
            Self::DenseVectorF32 | Self::DenseVectorF16 | Self::DenseVectorI8
        )
    }

    /// Returns whether this type is a supported sparse vector type.
    pub const fn is_sparse_vector(&self) -> bool {
        matches!(self, Self::SparseVectorF32 | Self::SparseVectorF16)
    }

    /// Returns whether a field of this type requires a fixed dimension.
    pub const fn requires_dimension(&self) -> bool {
        self.is_vector()
    }
}

/// The distance or similarity metric used for vector search.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MetricType {
    /// Euclidean distance.
    L2,
    /// Inner-product similarity.
    InnerProduct,
    /// Cosine similarity.
    Cosine,
}

// The newtype exists to keep internal identifiers from being confused with other u64 values;
// the monotonic, no-reuse allocation policy is enforced by the storage layer, not by DocId::new.
/// A globally unique internal document identifier.
///
/// Identifiers are allocated monotonically and never reused. [`DocId::new`] does not validate its
/// argument.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct DocId(u64);

impl DocId {
    /// Wraps an internal identifier value.
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the underlying integer value.
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl From<DocId> for u64 {
    fn from(value: DocId) -> Self {
        value.get()
    }
}

impl fmt::Display for DocId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}
