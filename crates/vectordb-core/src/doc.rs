use std::collections::BTreeMap;

use crate::{DataType, Error, F16, FieldSchema, Result, Schema, schema::PRIMARY_KEY_FIELD_NAME};

// Doc is deliberately an unvalidated working buffer, unlike FieldSchema/Schema whose existence
// proves validity: requiring a schema at set time would force callers to thread &Schema through
// every assembly site, and validate() gives one deterministic checkpoint instead.
/// A document assembled from typed field values.
///
/// Setters record values without a schema so documents can be assembled incrementally. Call
/// [`Doc::validate`] before insertion to enforce field presence, nullability, logical types, vector
/// dimensions, finite floating-point values, and configured limits.
///
/// Both sparse-vector setters share one contract: indices and values are retained exactly as
/// supplied, and [`Doc::validate`] rejects mismatched sequence lengths; unsorted, duplicate, or
/// out-of-bounds indices; non-finite values; and vectors that exceed the configured entry limit.
/// Validation never sorts, deduplicates, or otherwise mutates the caller's values. Empty index and
/// value sequences are valid, and entries whose values are `0.0` or `-0.0` are retained and counted
/// toward the entry limit rather than being removed.
///
/// Equality follows the stored floating-point values, so a document containing NaN is unequal to
/// itself.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Doc {
    values: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, PartialEq)]
enum Value {
    Null,
    Bool(bool),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    String(String),
    Binary(Vec<u8>),
    BoolArray(Vec<bool>),
    I32Array(Vec<i32>),
    I64Array(Vec<i64>),
    U32Array(Vec<u32>),
    U64Array(Vec<u64>),
    F32Array(Vec<f32>),
    F64Array(Vec<f64>),
    StringArray(Vec<String>),
    BinaryArray(Vec<Vec<u8>>),
    DenseVectorF32(Vec<f32>),
    DenseVectorF16(Vec<F16>),
    DenseVectorI8(Vec<i8>),
    SparseVectorF32 { indices: Vec<u32>, values: Vec<f32> },
    SparseVectorF16 { indices: Vec<u32>, values: Vec<F16> },
}

impl Value {
    const fn data_type(&self) -> Option<DataType> {
        match self {
            Self::Null => None,
            Self::Bool(_) => Some(DataType::Bool),
            Self::I32(_) => Some(DataType::I32),
            Self::I64(_) => Some(DataType::I64),
            Self::U32(_) => Some(DataType::U32),
            Self::U64(_) => Some(DataType::U64),
            Self::F32(_) => Some(DataType::F32),
            Self::F64(_) => Some(DataType::F64),
            Self::String(_) => Some(DataType::String),
            Self::Binary(_) => Some(DataType::Binary),
            Self::BoolArray(_) => Some(DataType::BoolArray),
            Self::I32Array(_) => Some(DataType::I32Array),
            Self::I64Array(_) => Some(DataType::I64Array),
            Self::U32Array(_) => Some(DataType::U32Array),
            Self::U64Array(_) => Some(DataType::U64Array),
            Self::F32Array(_) => Some(DataType::F32Array),
            Self::F64Array(_) => Some(DataType::F64Array),
            Self::StringArray(_) => Some(DataType::StringArray),
            Self::BinaryArray(_) => Some(DataType::BinaryArray),
            Self::DenseVectorF32(_) => Some(DataType::DenseVectorF32),
            Self::DenseVectorF16(_) => Some(DataType::DenseVectorF16),
            Self::DenseVectorI8(_) => Some(DataType::DenseVectorI8),
            Self::SparseVectorF32 { .. } => Some(DataType::SparseVectorF32),
            Self::SparseVectorF16 { .. } => Some(DataType::SparseVectorF16),
        }
    }
}

macro_rules! scalar_accessors {
    ($setter:ident, $getter:ident, $variant:ident, $ty:ty, $setter_doc:literal, $getter_doc:literal) => {
        #[doc = $setter_doc]
        pub fn $setter(&mut self, name: impl Into<String>, value: $ty) -> &mut Self {
            self.insert(name, Value::$variant(value))
        }

        #[doc = $getter_doc]
        ///
        /// Missing and explicitly null values both return `Ok(None)`; use [`Doc::contains_field`] and
        /// [`Doc::is_null`] to distinguish them.
        ///
        /// # Errors
        ///
        /// Returns [`Error::InvalidArgument`] when the stored non-null value has a different type
        /// than the getter expects.
        pub fn $getter(&self, name: &str) -> Result<Option<$ty>> {
            match self.values.get(name) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::$variant(value)) => Ok(Some(*value)),
                Some(value) => Err(type_mismatch(name, DataType::$variant, value)),
            }
        }
    };
}

macro_rules! slice_accessors {
    ($setter:ident, $getter:ident, $variant:ident, $ty:ty, $setter_doc:literal, $getter_doc:literal) => {
        #[doc = $setter_doc]
        pub fn $setter(
            &mut self,
            name: impl Into<String>,
            values: impl Into<Vec<$ty>>,
        ) -> &mut Self {
            self.insert(name, Value::$variant(values.into()))
        }

        #[doc = $getter_doc]
        ///
        /// Missing and explicitly null values both return `Ok(None)`; use [`Doc::contains_field`] and
        /// [`Doc::is_null`] to distinguish them.
        ///
        /// # Errors
        ///
        /// Returns [`Error::InvalidArgument`] when the stored non-null value has a different type
        /// than the getter expects.
        pub fn $getter(&self, name: &str) -> Result<Option<&[$ty]>> {
            match self.values.get(name) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::$variant(values)) => Ok(Some(values)),
                Some(value) => Err(type_mismatch(name, DataType::$variant, value)),
            }
        }
    };
}

impl Doc {
    /// Creates an empty document.
    pub const fn new() -> Self {
        Self {
            values: BTreeMap::new(),
        }
    }

    /// Records an explicit null value for a field.
    pub fn set_null(&mut self, name: impl Into<String>) -> &mut Self {
        self.insert(name, Value::Null)
    }

    /// Returns `true` if the document contains an entry for `name`, including an explicit null.
    pub fn contains_field(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// Returns `true` if the document stores an explicit null for `name`.
    ///
    /// A missing field returns `false`.
    pub fn is_null(&self, name: &str) -> bool {
        matches!(self.values.get(name), Some(Value::Null))
    }

    scalar_accessors!(
        set_bool,
        get_bool,
        Bool,
        bool,
        "Sets a Boolean field value.",
        "Returns a Boolean field value."
    );
    scalar_accessors!(
        set_i32,
        get_i32,
        I32,
        i32,
        "Sets a signed 32-bit integer field value.",
        "Returns a signed 32-bit integer field value."
    );
    scalar_accessors!(
        set_i64,
        get_i64,
        I64,
        i64,
        "Sets a signed 64-bit integer field value.",
        "Returns a signed 64-bit integer field value."
    );
    scalar_accessors!(
        set_u32,
        get_u32,
        U32,
        u32,
        "Sets an unsigned 32-bit integer field value.",
        "Returns an unsigned 32-bit integer field value."
    );
    scalar_accessors!(
        set_u64,
        get_u64,
        U64,
        u64,
        "Sets an unsigned 64-bit integer field value.",
        "Returns an unsigned 64-bit integer field value."
    );
    scalar_accessors!(
        set_f32,
        get_f32,
        F32,
        f32,
        "Sets a single-precision floating-point field value; validation rejects non-finite values.",
        "Returns a single-precision floating-point field value."
    );
    scalar_accessors!(
        set_f64,
        get_f64,
        F64,
        f64,
        "Sets a double-precision floating-point field value; validation rejects non-finite values.",
        "Returns a double-precision floating-point field value."
    );

    /// Sets a UTF-8 string field value.
    pub fn set_string(&mut self, name: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.insert(name, Value::String(value.into()))
    }

    /// Returns a UTF-8 string field value.
    ///
    /// Missing and explicitly null values both return `Ok(None)`; use [`Doc::contains_field`] and
    /// [`Doc::is_null`] to distinguish them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when the stored non-null value has a different type than
    /// the getter expects.
    pub fn get_string(&self, name: &str) -> Result<Option<&str>> {
        match self.values.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(value)) => Ok(Some(value)),
            Some(value) => Err(type_mismatch(name, DataType::String, value)),
        }
    }

    slice_accessors!(
        set_binary,
        get_binary,
        Binary,
        u8,
        "Sets an uninterpreted byte-string field value.",
        "Returns an uninterpreted byte-string field value."
    );
    slice_accessors!(
        set_bool_array,
        get_bool_array,
        BoolArray,
        bool,
        "Sets a Boolean array field value.",
        "Returns a Boolean array field value."
    );
    slice_accessors!(
        set_i32_array,
        get_i32_array,
        I32Array,
        i32,
        "Sets a signed 32-bit integer array field value.",
        "Returns a signed 32-bit integer array field value."
    );
    slice_accessors!(
        set_i64_array,
        get_i64_array,
        I64Array,
        i64,
        "Sets a signed 64-bit integer array field value.",
        "Returns a signed 64-bit integer array field value."
    );
    slice_accessors!(
        set_u32_array,
        get_u32_array,
        U32Array,
        u32,
        "Sets an unsigned 32-bit integer array field value.",
        "Returns an unsigned 32-bit integer array field value."
    );
    slice_accessors!(
        set_u64_array,
        get_u64_array,
        U64Array,
        u64,
        "Sets an unsigned 64-bit integer array field value.",
        "Returns an unsigned 64-bit integer array field value."
    );
    slice_accessors!(
        set_f32_array,
        get_f32_array,
        F32Array,
        f32,
        "Sets a single-precision floating-point array field value; validation rejects non-finite elements.",
        "Returns a single-precision floating-point array field value."
    );
    slice_accessors!(
        set_f64_array,
        get_f64_array,
        F64Array,
        f64,
        "Sets a double-precision floating-point array field value; validation rejects non-finite elements.",
        "Returns a double-precision floating-point array field value."
    );

    /// Sets a UTF-8 string array field value.
    pub fn set_string_array(
        &mut self,
        name: impl Into<String>,
        values: impl Into<Vec<String>>,
    ) -> &mut Self {
        self.insert(name, Value::StringArray(values.into()))
    }

    /// Returns a UTF-8 string array field value.
    ///
    /// Missing and explicitly null values both return `Ok(None)`; use [`Doc::contains_field`] and
    /// [`Doc::is_null`] to distinguish them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when the stored non-null value has a different type than
    /// the getter expects.
    pub fn get_string_array(&self, name: &str) -> Result<Option<&[String]>> {
        match self.values.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::StringArray(values)) => Ok(Some(values)),
            Some(value) => Err(type_mismatch(name, DataType::StringArray, value)),
        }
    }

    /// Sets an array of uninterpreted byte strings.
    pub fn set_binary_array(
        &mut self,
        name: impl Into<String>,
        values: impl Into<Vec<Vec<u8>>>,
    ) -> &mut Self {
        self.insert(name, Value::BinaryArray(values.into()))
    }

    /// Returns an array of uninterpreted byte strings.
    ///
    /// Missing and explicitly null values both return `Ok(None)`; use [`Doc::contains_field`] and
    /// [`Doc::is_null`] to distinguish them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when the stored non-null value has a different type than
    /// the getter expects.
    pub fn get_binary_array(&self, name: &str) -> Result<Option<&[Vec<u8>]>> {
        match self.values.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::BinaryArray(values)) => Ok(Some(values)),
            Some(value) => Err(type_mismatch(name, DataType::BinaryArray, value)),
        }
    }

    slice_accessors!(
        set_dense_vector_f32,
        get_dense_vector_f32,
        DenseVectorF32,
        f32,
        "Sets a dense single-precision vector field value.",
        "Returns a dense single-precision vector field value."
    );
    slice_accessors!(
        set_dense_vector_f16,
        get_dense_vector_f16,
        DenseVectorF16,
        F16,
        "Sets a dense half-precision vector field value.",
        "Returns a dense half-precision vector field value."
    );
    slice_accessors!(
        set_dense_vector_i8,
        get_dense_vector_i8,
        DenseVectorI8,
        i8,
        "Sets a dense signed 8-bit integer vector field value.",
        "Returns a dense signed 8-bit integer vector field value."
    );

    /// Sets a sparse single-precision vector field value.
    ///
    /// The retention and validation rules shared by both sparse-vector setters are documented on
    /// [`Doc`].
    pub fn set_sparse_vector_f32(
        &mut self,
        name: impl Into<String>,
        indices: impl Into<Vec<u32>>,
        values: impl Into<Vec<f32>>,
    ) -> &mut Self {
        self.insert(
            name,
            Value::SparseVectorF32 {
                indices: indices.into(),
                values: values.into(),
            },
        )
    }

    /// Returns a sparse single-precision vector as `(indices, values)`.
    ///
    /// Missing and explicitly null values both return `Ok(None)`; use [`Doc::contains_field`] and
    /// [`Doc::is_null`] to distinguish them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when the stored non-null value has a different type than
    /// the getter expects.
    pub fn get_sparse_vector_f32(&self, name: &str) -> Result<Option<(&[u32], &[f32])>> {
        match self.values.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::SparseVectorF32 { indices, values }) => Ok(Some((indices, values))),
            Some(value) => Err(type_mismatch(name, DataType::SparseVectorF32, value)),
        }
    }

    /// Sets a sparse half-precision vector field value.
    ///
    /// The retention and validation rules shared by both sparse-vector setters are documented on
    /// [`Doc`].
    pub fn set_sparse_vector_f16(
        &mut self,
        name: impl Into<String>,
        indices: impl Into<Vec<u32>>,
        values: impl Into<Vec<F16>>,
    ) -> &mut Self {
        self.insert(
            name,
            Value::SparseVectorF16 {
                indices: indices.into(),
                values: values.into(),
            },
        )
    }

    /// Returns a sparse half-precision vector as `(indices, values)`.
    ///
    /// Missing and explicitly null values both return `Ok(None)`; use [`Doc::contains_field`] and
    /// [`Doc::is_null`] to distinguish them.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when the stored non-null value has a different type than
    /// the getter expects.
    pub fn get_sparse_vector_f16(&self, name: &str) -> Result<Option<(&[u32], &[F16])>> {
        match self.values.get(name) {
            None | Some(Value::Null) => Ok(None),
            Some(Value::SparseVectorF16 { indices, values }) => Ok(Some((indices, values))),
            Some(value) => Err(type_mismatch(name, DataType::SparseVectorF16, value)),
        }
    }

    /// Validates every document value against `schema` without mutating the document.
    ///
    /// Validation checks `F32` and `F64` scalars, every element of `F32` and `F64` arrays, and all
    /// floating-point vector components for finiteness.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if the document contains an unknown field; omits or stores
    /// null in a non-nullable field; stores a value with the wrong logical type or vector shape;
    /// contains a value that exceeds a configured limit; or contains a non-finite floating-point value.
    pub fn validate(&self, schema: &Schema) -> Result<()> {
        if self.values.contains_key(PRIMARY_KEY_FIELD_NAME) {
            return Err(Error::invalid_argument(
                "`pk` is the implicit primary key and cannot be set on a document",
            ));
        }

        for (name, value) in &self.values {
            let field = schema.field(name).ok_or_else(|| {
                Error::invalid_argument(format!("document contains unknown field `{name}`"))
            })?;
            validate_value(name, value, field, schema)?;
        }

        for field in schema.fields() {
            if !field.is_nullable()
                && matches!(self.values.get(field.name()), None | Some(Value::Null))
            {
                return Err(Error::invalid_argument(format!(
                    "non-nullable field `{}` is missing or null",
                    field.name()
                )));
            }
        }

        Ok(())
    }

    fn insert(&mut self, name: impl Into<String>, value: Value) -> &mut Self {
        let _ = self.values.insert(name.into(), value);
        self
    }
}

fn type_mismatch(name: &str, expected: DataType, actual: &Value) -> Error {
    let actual = actual.data_type().map_or_else(
        || {
            // Callers handle null before this point, so this branch is unreachable today; the
            // "null" description is kept as a guard in case that ordering ever changes.
            "null".to_owned()
        },
        |data_type| format!("{data_type:?}"),
    );
    Error::invalid_argument(format!(
        "field `{name}` has type {actual}, expected {expected:?}"
    ))
}

fn validate_value(name: &str, value: &Value, field: &FieldSchema, schema: &Schema) -> Result<()> {
    if matches!(value, Value::Null) {
        return if field.is_nullable() {
            Ok(())
        } else {
            Err(Error::invalid_argument(format!(
                "non-nullable field `{name}` cannot be null"
            )))
        };
    }

    if value.data_type() != Some(field.data_type()) {
        return Err(type_mismatch(name, field.data_type(), value));
    }

    // Intentionally stricter than the roadmap's vector-only minimum: scalar floats and arrays must
    // also contain only finite values.
    match value {
        Value::F32(value) => ensure_finite(name, value.is_finite()),
        Value::F64(value) => ensure_finite(name, value.is_finite()),
        Value::F32Array(values) => {
            ensure_finite(name, values.iter().all(|value| value.is_finite()))
        }
        Value::F64Array(values) => {
            ensure_finite(name, values.iter().all(|value| value.is_finite()))
        }
        Value::DenseVectorF32(values) => {
            validate_dense_dimension(name, values.len(), vector_dimension(field)?)?;
            ensure_finite(name, values.iter().all(|value| value.is_finite()))
        }
        Value::DenseVectorF16(values) => {
            validate_dense_dimension(name, values.len(), vector_dimension(field)?)?;
            ensure_finite(name, values.iter().all(|value| value.is_finite()))
        }
        Value::DenseVectorI8(values) => {
            validate_dense_dimension(name, values.len(), vector_dimension(field)?)
        }
        Value::SparseVectorF32 { indices, values } => {
            validate_sparse_shape(
                name,
                indices,
                values.len(),
                vector_dimension(field)?,
                schema.limits().max_sparse_vector_entries,
            )?;
            ensure_finite(name, values.iter().all(|value| value.is_finite()))
        }
        Value::SparseVectorF16 { indices, values } => {
            validate_sparse_shape(
                name,
                indices,
                values.len(),
                vector_dimension(field)?,
                schema.limits().max_sparse_vector_entries,
            )?;
            ensure_finite(name, values.iter().all(|value| value.is_finite()))
        }
        Value::Null
        | Value::Bool(_)
        | Value::I32(_)
        | Value::I64(_)
        | Value::U32(_)
        | Value::U64(_)
        | Value::String(_)
        | Value::Binary(_)
        | Value::BoolArray(_)
        | Value::I32Array(_)
        | Value::I64Array(_)
        | Value::U32Array(_)
        | Value::U64Array(_)
        | Value::StringArray(_)
        | Value::BinaryArray(_) => Ok(()),
    }
}

fn vector_dimension(field: &FieldSchema) -> Result<usize> {
    field.dimension().ok_or_else(|| {
        Error::internal(format!(
            "validated vector field `{}` has no dimension",
            field.name()
        ))
    })
}

fn ensure_finite(name: &str, finite: bool) -> Result<()> {
    if finite {
        Ok(())
    } else {
        Err(Error::invalid_argument(format!(
            "field `{name}` contains a non-finite floating-point value"
        )))
    }
}

fn validate_dense_dimension(name: &str, actual: usize, expected: usize) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::invalid_argument(format!(
            "dense vector field `{name}` has dimension {actual}, expected {expected}"
        )))
    }
}

fn validate_sparse_shape(
    name: &str,
    indices: &[u32],
    values_len: usize,
    dimension: usize,
    max_entries: usize,
) -> Result<()> {
    if indices.len() != values_len {
        return Err(Error::invalid_argument(format!(
            "sparse vector field `{name}` has {} indices but {values_len} values",
            indices.len()
        )));
    }
    if indices.len() > max_entries {
        return Err(Error::invalid_argument(format!(
            "sparse vector field `{name}` has {} entries, exceeding limit {max_entries}",
            indices.len()
        )));
    }
    if !indices.windows(2).all(|pair| pair[0] < pair[1]) {
        return Err(Error::invalid_argument(format!(
            "sparse vector field `{name}` indices must be strictly increasing and unique"
        )));
    }
    if indices
        .iter()
        .any(|&index| u128::from(index) >= dimension as u128)
    {
        return Err(Error::invalid_argument(format!(
            "sparse vector field `{name}` contains an index outside dimension {dimension}"
        )));
    }

    Ok(())
}
