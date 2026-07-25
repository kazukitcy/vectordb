use std::{collections::HashSet, error::Error as _};

use vectordb_core::{DataType, Doc, DocId, Error, F16, FieldSchema, Limits, MetricType, Schema};

fn field(name: &str, data_type: DataType) -> FieldSchema {
    FieldSchema::builder(name, data_type).build().unwrap()
}

fn vector_field(name: &str, data_type: DataType, dimension: usize) -> FieldSchema {
    FieldSchema::builder(name, data_type)
        .dimension(dimension)
        .build()
        .unwrap()
}

fn one_field_schema(field: FieldSchema) -> Schema {
    Schema::builder().field(field).build().unwrap()
}

fn assert_invalid_argument(error: &Error) {
    assert!(
        matches!(error, Error::InvalidArgument { .. }),
        "expected InvalidArgument, got {error:?}"
    );
}

type GetterProbe = fn(&Doc, &str) -> vectordb_core::Result<()>;
type GetterPresenceProbe = fn(&Doc, &str) -> vectordb_core::Result<bool>;

fn assert_getter_type_mismatches(doc: &Doc, cases: &[(&str, GetterProbe)]) {
    for (getter_name, getter) in cases {
        let error = getter(doc, "value").unwrap_err();
        assert!(
            matches!(error, Error::InvalidArgument { .. }),
            "{getter_name} returned {error:?}"
        );
    }
}

#[test]
fn core_types_expose_the_m0_contract() {
    let scalar_types = [
        DataType::Bool,
        DataType::I32,
        DataType::I64,
        DataType::U32,
        DataType::U64,
        DataType::F32,
        DataType::F64,
        DataType::String,
        DataType::Binary,
    ];
    let array_types = [
        DataType::BoolArray,
        DataType::I32Array,
        DataType::I64Array,
        DataType::U32Array,
        DataType::U64Array,
        DataType::F32Array,
        DataType::F64Array,
        DataType::StringArray,
        DataType::BinaryArray,
    ];
    let dense_types = [
        DataType::DenseVectorF32,
        DataType::DenseVectorF16,
        DataType::DenseVectorI8,
    ];
    let sparse_types = [DataType::SparseVectorF32, DataType::SparseVectorF16];

    assert!(scalar_types.iter().all(|data_type| !data_type.is_vector()));
    assert!(array_types.iter().all(|data_type| !data_type.is_vector()));
    assert!(dense_types.iter().all(DataType::is_dense_vector));
    assert!(sparse_types.iter().all(DataType::is_sparse_vector));
    assert!(
        dense_types
            .iter()
            .chain(sparse_types.iter())
            .all(DataType::requires_dimension)
    );

    assert_eq!(MetricType::L2, MetricType::L2);
    assert_ne!(MetricType::InnerProduct, MetricType::Cosine);

    let id = DocId::new(42);
    let copied = id;
    assert_eq!(id.get(), 42);
    assert_eq!(copied, id);
    assert_eq!(u64::from(id), 42);
    assert!(DocId::new(41) < id);
    assert_eq!(format!("{id:?}"), "DocId(42)");
    assert!(HashSet::from([id]).contains(&id));
}

#[test]
fn limits_have_configurable_documented_defaults() {
    let mut limits = Limits::default();
    assert_eq!(limits.max_vector_dimension, 16_384);
    assert_eq!(limits.max_sparse_vector_entries, 65_536);
    assert_eq!(limits.max_vector_fields, 4);
    assert_eq!(limits.max_top_k, 65_536);
    assert_eq!(limits.max_batch_write_documents, 10_000);

    limits.max_vector_dimension = 8;
    limits.max_sparse_vector_entries = 3;
    assert_eq!(limits.max_vector_dimension, 8);
    assert_eq!(limits.max_sparse_vector_entries, 3);
}

#[test]
fn io_errors_keep_the_std_error_as_their_source() {
    let error = Error::io(std::io::Error::new(
        std::io::ErrorKind::UnexpectedEof,
        "short read",
    ));

    assert!(matches!(error, Error::Io { .. }));
    assert_eq!(error.source().unwrap().to_string(), "short read");
    assert_eq!(error.message(), None);
    assert!(error.to_string().starts_with("I/O error"));

    let converted = Error::from(std::io::Error::other("converted"));
    assert!(matches!(converted, Error::Io { .. }));
}

#[test]
fn error_exposes_every_m0_category_through_constructors() {
    macro_rules! assert_message_error {
        ($error:expr, $pattern:pat, $message:literal) => {{
            let error = $error;
            assert_eq!(error.message(), Some($message));
            assert!(!error.to_string().is_empty());
            assert!(matches!(error, $pattern));
        }};
    }

    assert_message_error!(
        Error::invalid_argument("invalid"),
        Error::InvalidArgument { .. },
        "invalid"
    );
    assert_message_error!(
        Error::not_found("missing"),
        Error::NotFound { .. },
        "missing"
    );
    assert_message_error!(
        Error::already_exists("duplicate"),
        Error::AlreadyExists { .. },
        "duplicate"
    );
    assert_message_error!(
        Error::corruption("corrupt"),
        Error::Corruption { .. },
        "corrupt"
    );
    assert_message_error!(
        Error::unsupported("unsupported"),
        Error::Unsupported { .. },
        "unsupported"
    );
    assert_message_error!(
        Error::internal("internal"),
        Error::Internal { .. },
        "internal"
    );
}

#[test]
fn f16_preserves_ieee_754_bits_and_partial_comparison_semantics() {
    for bits in 0..=u16::MAX {
        assert_eq!(F16::from_bits(bits).to_bits(), bits);
    }

    assert_eq!(F16::from_bits(0x3c00).to_f32().to_bits(), 1.0_f32.to_bits());
    assert!(F16::from_f32(1.5).is_finite());
    assert!(!F16::from_bits(0x7e00).is_finite());
    assert!(!F16::from_bits(0x7c00).is_finite());
    assert!(!F16::from_bits(0xfc00).is_finite());
    assert_eq!(F16::default().to_bits(), 0);

    let nan_a = F16::from_bits(0x7e01);
    let nan_b = F16::from_bits(0x7e01);
    assert_ne!(nan_a, nan_b);
    assert_eq!(nan_a.partial_cmp(&nan_b), None);
    assert_eq!(F16::from_bits(0x8000), F16::from_bits(0x0000));
}

#[test]
fn f16_from_f32_rounds_to_nearest_ties_to_even() {
    // Each midpoint sits exactly between two adjacent f16 values; ties-to-even keeps the even
    // neighbor, which distinguishes it from truncation and from ties-away-from-zero.
    let midpoint_below_odd = f32::from_bits(0x3F80_1000); // 1 + 2^-11, between 0x3c00 and 0x3c01
    let midpoint_above_odd = f32::from_bits(0x3F80_3000); // 1 + 3 * 2^-11, between 0x3c01 and 0x3c02
    assert_eq!(F16::from_f32(midpoint_below_odd).to_bits(), 0x3c00);
    assert_eq!(F16::from_f32(midpoint_above_odd).to_bits(), 0x3c02);

    for bits in 0..=u16::MAX {
        let value = F16::from_bits(bits);
        if value.is_finite() {
            assert_eq!(F16::from_f32(value.to_f32()).to_bits(), bits);
        }
    }
}

#[test]
fn schema_builds_scalar_and_vector_fields() {
    let title = FieldSchema::builder("title", DataType::String)
        .nullable(true)
        .build()
        .unwrap();
    let embedding = vector_field("embedding", DataType::DenseVectorF32, 3);

    let schema = Schema::builder()
        .field(title)
        .field(embedding)
        .build()
        .unwrap();

    assert_eq!(schema.fields().len(), 2);
    let title = schema.field("title").unwrap();
    assert_eq!(title.name(), "title");
    assert_eq!(title.data_type(), DataType::String);
    assert!(title.is_nullable());
    assert_eq!(title.dimension(), None);
    assert_eq!(schema.field("embedding").unwrap().dimension(), Some(3));
    assert_eq!(schema.limits(), &Limits::default());
}

#[test]
fn schema_rejects_invalid_and_reserved_field_names() {
    for name in ["", "9field", "has-dash", "white space", "éclair", "pk"] {
        let error = FieldSchema::builder(name, DataType::String)
            .build()
            .unwrap_err();
        assert_invalid_argument(&error);
    }

    for name in ["a", "_internal", "field_9"] {
        FieldSchema::builder(name, DataType::String)
            .build()
            .unwrap();
    }
}

#[test]
fn schema_rejects_duplicate_fields() {
    let error = Schema::builder()
        .field(field("title", DataType::String))
        .field(field("title", DataType::String))
        .build()
        .unwrap_err();

    assert_invalid_argument(&error);
}

#[test]
fn field_schema_enforces_dimension_contract() {
    let missing = FieldSchema::builder("embedding", DataType::DenseVectorF32)
        .build()
        .unwrap_err();
    assert_invalid_argument(&missing);

    let scalar_dimension = FieldSchema::builder("score", DataType::F32)
        .dimension(3)
        .build()
        .unwrap_err();
    assert_invalid_argument(&scalar_dimension);

    let zero = FieldSchema::builder("embedding", DataType::DenseVectorF32)
        .dimension(0)
        .build()
        .unwrap_err();
    assert_invalid_argument(&zero);

    FieldSchema::builder("embedding", DataType::DenseVectorF32)
        .dimension(Limits::default().max_vector_dimension)
        .build()
        .unwrap();

    let too_large = FieldSchema::builder("embedding", DataType::DenseVectorF32)
        .dimension(Limits::default().max_vector_dimension + 1)
        .build()
        .unwrap_err();
    assert_invalid_argument(&too_large);
}

#[cfg(target_pointer_width = "64")]
#[test]
fn sparse_field_dimension_is_bounded_by_u32_coordinates() {
    let largest_dimension = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
    let unrepresentable_dimension = usize::try_from(u64::from(u32::MAX) + 2).unwrap();
    let mut limits = Limits::default();
    limits.max_vector_dimension = unrepresentable_dimension;

    for data_type in [DataType::SparseVectorF32, DataType::SparseVectorF16] {
        FieldSchema::builder("sparse", data_type)
            .dimension(largest_dimension)
            .build_with_limits(&limits)
            .unwrap();

        let error = FieldSchema::builder("sparse", data_type)
            .dimension(unrepresentable_dimension)
            .build_with_limits(&limits)
            .unwrap_err();
        assert_invalid_argument(&error);
    }
}

#[test]
fn sparse_field_dimension_obeys_the_configured_limit() {
    let mut limits = Limits::default();
    limits.max_vector_dimension = 4;

    for data_type in [DataType::SparseVectorF32, DataType::SparseVectorF16] {
        FieldSchema::builder("sparse", data_type)
            .dimension(limits.max_vector_dimension)
            .build_with_limits(&limits)
            .unwrap();

        let error = FieldSchema::builder("sparse", data_type)
            .dimension(limits.max_vector_dimension + 1)
            .build_with_limits(&limits)
            .unwrap_err();
        assert_invalid_argument(&error);
    }
}

#[test]
fn schema_uses_custom_limits_and_revalidates_fields() {
    let mut permissive = Limits::default();
    permissive.max_vector_dimension += 1;
    let oversized_for_default = FieldSchema::builder("embedding", DataType::DenseVectorF32)
        .dimension(permissive.max_vector_dimension)
        .build_with_limits(&permissive)
        .unwrap();

    let default_error = Schema::builder()
        .field(oversized_for_default.clone())
        .build()
        .unwrap_err();
    assert_invalid_argument(&default_error);

    Schema::builder()
        .limits(permissive)
        .field(oversized_for_default)
        .build()
        .unwrap();
}

#[test]
fn schema_enforces_the_vector_field_limit() {
    let mut builder = Schema::builder();
    for index in 0..Limits::default().max_vector_fields {
        builder = builder.field(vector_field(
            &format!("vector_{index}"),
            DataType::DenseVectorF32,
            1,
        ));
    }
    builder.build().unwrap();

    let mut too_many = Schema::builder();
    for index in 0..=Limits::default().max_vector_fields {
        too_many = too_many.field(vector_field(
            &format!("vector_{index}"),
            DataType::DenseVectorF32,
            1,
        ));
    }
    assert_invalid_argument(&too_many.build().unwrap_err());
}

#[test]
fn sparse_fields_count_toward_the_vector_field_limit() {
    for sparse_type in [DataType::SparseVectorF32, DataType::SparseVectorF16] {
        let mut limits = Limits::default();
        limits.max_vector_fields = 1;

        let error = Schema::builder()
            .limits(limits)
            .field(vector_field("dense", DataType::DenseVectorF32, 1))
            .field(vector_field("sparse", sparse_type, 1))
            .build()
            .unwrap_err();
        assert_invalid_argument(&error);
    }
}

#[test]
fn doc_round_trips_every_scalar_and_array_type() {
    let fields = [
        field("bool", DataType::Bool),
        field("i32", DataType::I32),
        field("i64", DataType::I64),
        field("u32", DataType::U32),
        field("u64", DataType::U64),
        field("f32", DataType::F32),
        field("f64", DataType::F64),
        field("string", DataType::String),
        field("binary", DataType::Binary),
        field("bool_array", DataType::BoolArray),
        field("i32_array", DataType::I32Array),
        field("i64_array", DataType::I64Array),
        field("u32_array", DataType::U32Array),
        field("u64_array", DataType::U64Array),
        field("f32_array", DataType::F32Array),
        field("f64_array", DataType::F64Array),
        field("string_array", DataType::StringArray),
        field("binary_array", DataType::BinaryArray),
    ];
    let mut schema_builder = Schema::builder();
    for field in fields {
        schema_builder = schema_builder.field(field);
    }
    let schema = schema_builder.build().unwrap();

    let mut doc = Doc::new();
    doc.set_bool("bool", true)
        .set_i32("i32", -32)
        .set_i64("i64", -64)
        .set_u32("u32", 32)
        .set_u64("u64", 64)
        .set_f32("f32", 3.25)
        .set_f64("f64", 6.5)
        .set_string("string", "hello")
        .set_binary("binary", [1, 2, 3])
        .set_bool_array("bool_array", vec![true, false])
        .set_i32_array("i32_array", vec![-1, 2])
        .set_i64_array("i64_array", vec![-3, 4])
        .set_u32_array("u32_array", vec![5, 6])
        .set_u64_array("u64_array", vec![7, 8])
        .set_f32_array("f32_array", vec![1.5, 2.5])
        .set_f64_array("f64_array", vec![3.5, 4.5])
        .set_string_array("string_array", vec!["a".to_owned(), "b".to_owned()])
        .set_binary_array("binary_array", vec![vec![9], vec![10, 11]]);

    doc.validate(&schema).unwrap();
    schema.validate_doc(&doc).unwrap();
    assert_eq!(doc.get_bool("bool").unwrap(), Some(true));
    assert_eq!(doc.get_i32("i32").unwrap(), Some(-32));
    assert_eq!(doc.get_i64("i64").unwrap(), Some(-64));
    assert_eq!(doc.get_u32("u32").unwrap(), Some(32));
    assert_eq!(doc.get_u64("u64").unwrap(), Some(64));
    assert_eq!(doc.get_f32("f32").unwrap(), Some(3.25));
    assert_eq!(doc.get_f64("f64").unwrap(), Some(6.5));
    assert_eq!(doc.get_string("string").unwrap(), Some("hello"));
    assert_eq!(doc.get_binary("binary").unwrap(), Some(&[1, 2, 3][..]));
    assert_eq!(
        doc.get_bool_array("bool_array").unwrap(),
        Some(&[true, false][..])
    );
    assert_eq!(doc.get_i32_array("i32_array").unwrap(), Some(&[-1, 2][..]));
    assert_eq!(doc.get_i64_array("i64_array").unwrap(), Some(&[-3, 4][..]));
    assert_eq!(doc.get_u32_array("u32_array").unwrap(), Some(&[5, 6][..]));
    assert_eq!(doc.get_u64_array("u64_array").unwrap(), Some(&[7, 8][..]));
    assert_eq!(
        doc.get_f32_array("f32_array").unwrap(),
        Some(&[1.5, 2.5][..])
    );
    assert_eq!(
        doc.get_f64_array("f64_array").unwrap(),
        Some(&[3.5, 4.5][..])
    );
    assert_eq!(
        doc.get_string_array("string_array").unwrap().unwrap(),
        ["a", "b"]
    );
    assert_eq!(
        doc.get_binary_array("binary_array").unwrap().unwrap(),
        [vec![9], vec![10, 11]]
    );
}

#[test]
fn doc_round_trips_all_supported_vector_types() {
    let mut limits = Limits::default();
    limits.max_vector_fields = 5;
    let schema = Schema::builder()
        .limits(limits)
        .field(vector_field("dense_f32", DataType::DenseVectorF32, 2))
        .field(vector_field("dense_f16", DataType::DenseVectorF16, 2))
        .field(vector_field("dense_i8", DataType::DenseVectorI8, 2))
        .field(vector_field("sparse_f32", DataType::SparseVectorF32, 8))
        .field(vector_field("sparse_f16", DataType::SparseVectorF16, 8))
        .build()
        .unwrap();
    let half = [F16::from_f32(1.0), F16::from_f32(-2.0)];

    let mut doc = Doc::new();
    doc.set_dense_vector_f32("dense_f32", vec![1.0, -2.0])
        .set_dense_vector_f16("dense_f16", half)
        .set_dense_vector_i8("dense_i8", vec![1, -2])
        .set_sparse_vector_f32("sparse_f32", vec![1, 7], vec![0.5, -0.25])
        .set_sparse_vector_f16("sparse_f16", vec![0, 6], half);

    doc.validate(&schema).unwrap();
    assert_eq!(
        doc.get_dense_vector_f32("dense_f32").unwrap(),
        Some(&[1.0, -2.0][..])
    );
    assert_eq!(
        doc.get_dense_vector_f16("dense_f16").unwrap(),
        Some(&half[..])
    );
    assert_eq!(
        doc.get_dense_vector_i8("dense_i8").unwrap(),
        Some(&[1, -2][..])
    );
    assert_eq!(
        doc.get_sparse_vector_f32("sparse_f32").unwrap(),
        Some((&[1, 7][..], &[0.5, -0.25][..]))
    );
    assert_eq!(
        doc.get_sparse_vector_f16("sparse_f16").unwrap(),
        Some((&[0, 6][..], &half[..]))
    );
}

#[test]
fn doc_allows_nullable_missing_and_explicit_null_values() {
    let nullable = FieldSchema::builder("note", DataType::String)
        .nullable(true)
        .build()
        .unwrap();
    let schema = one_field_schema(nullable);

    let missing = Doc::new();
    missing.validate(&schema).unwrap();
    assert!(!missing.contains_field("note"));

    let mut explicit = Doc::new();
    explicit.set_null("note");
    explicit.validate(&schema).unwrap();
    assert!(explicit.contains_field("note"));
    assert!(explicit.is_null("note"));
    assert_eq!(explicit.get_string("note").unwrap(), None);
}

#[test]
fn doc_rejects_missing_and_null_non_nullable_fields() {
    let schema = one_field_schema(field("title", DataType::String));

    let missing = Doc::new().validate(&schema).unwrap_err();
    assert_invalid_argument(&missing);

    let mut explicit = Doc::new();
    explicit.set_null("title");
    let null = explicit.validate(&schema).unwrap_err();
    assert_invalid_argument(&null);
}

#[test]
fn doc_rejects_unknown_fields_and_type_mismatches() {
    let schema = one_field_schema(field("count", DataType::U64));

    let mut unknown = Doc::new();
    unknown.set_u64("count", 1).set_string("extra", "value");
    assert_invalid_argument(&unknown.validate(&schema).unwrap_err());

    let mut wrong_type = Doc::new();
    wrong_type.set_i64("count", 1);
    assert_invalid_argument(&wrong_type.validate(&schema).unwrap_err());
    assert_invalid_argument(&wrong_type.get_u64("count").unwrap_err());
}

#[test]
fn doc_rejects_the_implicit_primary_key_field_with_a_specific_error() {
    let schema = Schema::builder().build().unwrap();
    let mut doc = Doc::new();
    doc.set_string("pk", "external-key");

    let error = doc.validate(&schema).unwrap_err();
    assert_invalid_argument(&error);
    let message = error.message().unwrap();
    assert!(message.contains("`pk`"));
    assert!(message.contains("implicit primary key"));
    assert!(message.contains("cannot be set"));
}

#[test]
fn scalar_accessors_reject_every_wrong_stored_type() {
    let mut doc = Doc::new();
    doc.set_string("value", "wrong type");
    let cases: [(&str, GetterProbe); 7] = [
        ("get_bool", |doc, name| doc.get_bool(name).map(|_| ())),
        ("get_i32", |doc, name| doc.get_i32(name).map(|_| ())),
        ("get_i64", |doc, name| doc.get_i64(name).map(|_| ())),
        ("get_u32", |doc, name| doc.get_u32(name).map(|_| ())),
        ("get_u64", |doc, name| doc.get_u64(name).map(|_| ())),
        ("get_f32", |doc, name| doc.get_f32(name).map(|_| ())),
        ("get_f64", |doc, name| doc.get_f64(name).map(|_| ())),
    ];

    assert_getter_type_mismatches(&doc, &cases);
}

#[test]
fn slice_accessors_reject_every_wrong_stored_type() {
    let mut doc = Doc::new();
    doc.set_string("value", "wrong type");
    let cases: [(&str, GetterProbe); 11] = [
        ("get_binary", |doc, name| doc.get_binary(name).map(|_| ())),
        ("get_bool_array", |doc, name| {
            doc.get_bool_array(name).map(|_| ())
        }),
        ("get_i32_array", |doc, name| {
            doc.get_i32_array(name).map(|_| ())
        }),
        ("get_i64_array", |doc, name| {
            doc.get_i64_array(name).map(|_| ())
        }),
        ("get_u32_array", |doc, name| {
            doc.get_u32_array(name).map(|_| ())
        }),
        ("get_u64_array", |doc, name| {
            doc.get_u64_array(name).map(|_| ())
        }),
        ("get_f32_array", |doc, name| {
            doc.get_f32_array(name).map(|_| ())
        }),
        ("get_f64_array", |doc, name| {
            doc.get_f64_array(name).map(|_| ())
        }),
        ("get_dense_vector_f32", |doc, name| {
            doc.get_dense_vector_f32(name).map(|_| ())
        }),
        ("get_dense_vector_f16", |doc, name| {
            doc.get_dense_vector_f16(name).map(|_| ())
        }),
        ("get_dense_vector_i8", |doc, name| {
            doc.get_dense_vector_i8(name).map(|_| ())
        }),
    ];

    assert_getter_type_mismatches(&doc, &cases);
}

#[test]
fn handwritten_getters_reject_wrong_types_and_distinguish_missing_from_null() {
    let mut wrong_type = Doc::new();
    wrong_type.set_bool("value", true);
    let mismatch_cases: [(&str, GetterProbe); 5] = [
        ("get_string", |doc, name| doc.get_string(name).map(|_| ())),
        ("get_string_array", |doc, name| {
            doc.get_string_array(name).map(|_| ())
        }),
        ("get_binary_array", |doc, name| {
            doc.get_binary_array(name).map(|_| ())
        }),
        ("get_sparse_vector_f32", |doc, name| {
            doc.get_sparse_vector_f32(name).map(|_| ())
        }),
        ("get_sparse_vector_f16", |doc, name| {
            doc.get_sparse_vector_f16(name).map(|_| ())
        }),
    ];
    assert_getter_type_mismatches(&wrong_type, &mismatch_cases);

    let presence_cases: [(&str, GetterPresenceProbe); 5] = [
        ("get_string", |doc, name| {
            doc.get_string(name).map(|value| value.is_some())
        }),
        ("get_string_array", |doc, name| {
            doc.get_string_array(name).map(|value| value.is_some())
        }),
        ("get_binary_array", |doc, name| {
            doc.get_binary_array(name).map(|value| value.is_some())
        }),
        ("get_sparse_vector_f32", |doc, name| {
            doc.get_sparse_vector_f32(name).map(|value| value.is_some())
        }),
        ("get_sparse_vector_f16", |doc, name| {
            doc.get_sparse_vector_f16(name).map(|value| value.is_some())
        }),
    ];
    let missing = Doc::new();
    let mut explicit_null = Doc::new();
    explicit_null.set_null("value");

    for (getter_name, getter) in presence_cases {
        assert!(!getter(&missing, "value").unwrap(), "{getter_name}");
        assert!(!getter(&explicit_null, "value").unwrap(), "{getter_name}");
    }
    assert!(!missing.contains_field("value"));
    assert!(!missing.is_null("value"));
    assert!(explicit_null.contains_field("value"));
    assert!(explicit_null.is_null("value"));
}

#[test]
fn doc_rejects_dense_vector_dimension_mismatches() {
    let cases = [
        (DataType::DenseVectorF32, "f32"),
        (DataType::DenseVectorF16, "f16"),
        (DataType::DenseVectorI8, "i8"),
    ];

    for (data_type, name) in cases {
        let schema = one_field_schema(vector_field(name, data_type, 2));
        let mut doc = Doc::new();
        match data_type {
            DataType::DenseVectorF32 => {
                doc.set_dense_vector_f32(name, vec![1.0]);
            }
            DataType::DenseVectorF16 => {
                doc.set_dense_vector_f16(name, vec![F16::from_f32(1.0)]);
            }
            DataType::DenseVectorI8 => {
                doc.set_dense_vector_i8(name, vec![1]);
            }
            _ => unreachable!(),
        }
        assert_invalid_argument(&doc.validate(&schema).unwrap_err());

        match data_type {
            DataType::DenseVectorF32 => {
                doc.set_dense_vector_f32(name, vec![1.0, 2.0, 3.0]);
            }
            DataType::DenseVectorF16 => {
                doc.set_dense_vector_f16(name, vec![F16::from_f32(1.0); 3]);
            }
            DataType::DenseVectorI8 => {
                doc.set_dense_vector_i8(name, vec![1, 2, 3]);
            }
            _ => unreachable!(),
        }
        assert_invalid_argument(&doc.validate(&schema).unwrap_err());
    }
}

#[test]
fn doc_rejects_non_finite_scalar_array_dense_and_sparse_values() {
    let tests: Vec<(Schema, Doc)> = vec![
        {
            let mut doc = Doc::new();
            doc.set_f32("value", f32::NAN);
            (one_field_schema(field("value", DataType::F32)), doc)
        },
        {
            let mut doc = Doc::new();
            doc.set_f64("value", f64::INFINITY);
            (one_field_schema(field("value", DataType::F64)), doc)
        },
        {
            let mut doc = Doc::new();
            doc.set_f32_array("value", vec![f32::NEG_INFINITY]);
            (one_field_schema(field("value", DataType::F32Array)), doc)
        },
        {
            let mut doc = Doc::new();
            doc.set_f64_array("value", vec![f64::NAN]);
            (one_field_schema(field("value", DataType::F64Array)), doc)
        },
        {
            let mut doc = Doc::new();
            doc.set_dense_vector_f32("value", vec![f32::INFINITY]);
            (
                one_field_schema(vector_field("value", DataType::DenseVectorF32, 1)),
                doc,
            )
        },
        {
            let mut doc = Doc::new();
            doc.set_dense_vector_f16("value", vec![F16::from_f32(f32::NAN)]);
            (
                one_field_schema(vector_field("value", DataType::DenseVectorF16, 1)),
                doc,
            )
        },
        {
            let mut doc = Doc::new();
            doc.set_sparse_vector_f32("value", vec![0], vec![f32::NEG_INFINITY]);
            (
                one_field_schema(vector_field("value", DataType::SparseVectorF32, 1)),
                doc,
            )
        },
        {
            let mut doc = Doc::new();
            doc.set_sparse_vector_f16("value", vec![0], vec![F16::from_f32(f32::INFINITY)]);
            (
                one_field_schema(vector_field("value", DataType::SparseVectorF16, 1)),
                doc,
            )
        },
    ];

    for (schema, doc) in tests {
        assert_invalid_argument(&doc.validate(&schema).unwrap_err());
    }
}

#[test]
fn doc_rejects_sparse_length_order_duplicate_and_bounds_errors_without_normalizing() {
    let schema = one_field_schema(vector_field("sparse", DataType::SparseVectorF32, 4));

    for (indices, values) in [
        (vec![0, 1], vec![1.0]),
        (vec![2, 1], vec![1.0, 2.0]),
        (vec![1, 1], vec![1.0, 2.0]),
        (vec![0, 4], vec![1.0, 2.0]),
    ] {
        let mut doc = Doc::new();
        doc.set_sparse_vector_f32("sparse", indices.clone(), values);
        assert_invalid_argument(&doc.validate(&schema).unwrap_err());
        assert_eq!(
            doc.get_sparse_vector_f32("sparse").unwrap().unwrap().0,
            indices
        );
    }

    let f16_schema = one_field_schema(vector_field("sparse", DataType::SparseVectorF16, 4));
    let mut f16_doc = Doc::new();
    f16_doc.set_sparse_vector_f16(
        "sparse",
        vec![2, 1],
        vec![F16::from_f32(1.0), F16::from_f32(2.0)],
    );
    assert_invalid_argument(&f16_doc.validate(&f16_schema).unwrap_err());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn sparse_vector_indices_obey_the_u32_coordinate_boundary() {
    type SparseSetter = fn(&mut Doc, &str, u32);

    let largest_dimension = usize::try_from(u64::from(u32::MAX) + 1).unwrap();
    let out_of_bounds_dimension = usize::try_from(u64::from(u32::MAX)).unwrap();
    let mut limits = Limits::default();
    limits.max_vector_dimension = largest_dimension;
    let cases: [(DataType, SparseSetter); 2] = [
        (DataType::SparseVectorF32, |doc, name, index| {
            doc.set_sparse_vector_f32(name, vec![index], vec![1.0]);
        }),
        (DataType::SparseVectorF16, |doc, name, index| {
            doc.set_sparse_vector_f16(name, vec![index], vec![F16::from_f32(1.0)]);
        }),
    ];

    for (data_type, set_sparse) in cases {
        let boundary_field = FieldSchema::builder("sparse", data_type)
            .dimension(largest_dimension)
            .build_with_limits(&limits)
            .unwrap();
        let boundary_schema = Schema::builder()
            .limits(limits.clone())
            .field(boundary_field)
            .build()
            .unwrap();
        let mut boundary_doc = Doc::new();
        set_sparse(&mut boundary_doc, "sparse", u32::MAX);
        boundary_doc.validate(&boundary_schema).unwrap();

        let smaller_field = FieldSchema::builder("sparse", data_type)
            .dimension(out_of_bounds_dimension)
            .build_with_limits(&limits)
            .unwrap();
        let smaller_schema = Schema::builder()
            .limits(limits.clone())
            .field(smaller_field)
            .build()
            .unwrap();
        let mut out_of_bounds_doc = Doc::new();
        set_sparse(&mut out_of_bounds_doc, "sparse", u32::MAX);
        assert_invalid_argument(&out_of_bounds_doc.validate(&smaller_schema).unwrap_err());
    }
}

#[test]
fn doc_accepts_empty_sparse_vectors() {
    let schema = Schema::builder()
        .field(vector_field("sparse_f32", DataType::SparseVectorF32, 4))
        .field(vector_field("sparse_f16", DataType::SparseVectorF16, 4))
        .build()
        .unwrap();
    let mut doc = Doc::new();
    doc.set_sparse_vector_f32("sparse_f32", Vec::new(), Vec::new())
        .set_sparse_vector_f16("sparse_f16", Vec::new(), Vec::new());

    doc.validate(&schema).unwrap();
    assert_eq!(
        doc.get_sparse_vector_f32("sparse_f32").unwrap(),
        Some((&[][..], &[][..]))
    );
    assert_eq!(
        doc.get_sparse_vector_f16("sparse_f16").unwrap(),
        Some((&[][..], &[][..]))
    );
}

#[test]
fn doc_enforces_the_configured_sparse_entry_limit() {
    let mut limits = Limits::default();
    limits.max_sparse_vector_entries = 2;
    let field = FieldSchema::builder("sparse", DataType::SparseVectorF32)
        .dimension(4)
        .build_with_limits(&limits)
        .unwrap();
    let schema = Schema::builder()
        .limits(limits.clone())
        .field(field)
        .build()
        .unwrap();

    let mut at_limit = Doc::new();
    at_limit.set_sparse_vector_f32("sparse", vec![0, 3], vec![1.0, 2.0]);
    at_limit.validate(&schema).unwrap();

    let mut over_limit = Doc::new();
    over_limit.set_sparse_vector_f32("sparse", vec![0, 1, 2], vec![1.0, 2.0, 3.0]);
    assert_invalid_argument(&over_limit.validate(&schema).unwrap_err());

    let mut zeros_at_limit = Doc::new();
    zeros_at_limit.set_sparse_vector_f32("sparse", vec![0, 3], vec![0.0, -0.0]);
    zeros_at_limit.validate(&schema).unwrap();
    let (_, values) = zeros_at_limit
        .get_sparse_vector_f32("sparse")
        .unwrap()
        .unwrap();
    assert_eq!(values[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(values[1].to_bits(), (-0.0_f32).to_bits());

    let mut zeros_over_limit = Doc::new();
    zeros_over_limit.set_sparse_vector_f32("sparse", vec![0, 1, 2], vec![0.0, -0.0, 1.0]);
    assert_invalid_argument(&zeros_over_limit.validate(&schema).unwrap_err());

    let f16_field = FieldSchema::builder("sparse", DataType::SparseVectorF16)
        .dimension(4)
        .build_with_limits(&limits)
        .unwrap();
    let f16_schema = Schema::builder()
        .limits(limits)
        .field(f16_field)
        .build()
        .unwrap();

    let mut f16_zeros_at_limit = Doc::new();
    f16_zeros_at_limit.set_sparse_vector_f16(
        "sparse",
        vec![0, 3],
        vec![F16::from_bits(0x0000), F16::from_bits(0x8000)],
    );
    f16_zeros_at_limit.validate(&f16_schema).unwrap();
    let (_, f16_values) = f16_zeros_at_limit
        .get_sparse_vector_f16("sparse")
        .unwrap()
        .unwrap();
    assert_eq!(f16_values[0].to_bits(), 0x0000);
    assert_eq!(f16_values[1].to_bits(), 0x8000);

    let mut f16_zeros_over_limit = Doc::new();
    f16_zeros_over_limit.set_sparse_vector_f16(
        "sparse",
        vec![0, 1, 2],
        vec![
            F16::from_bits(0x0000),
            F16::from_f32(1.0),
            F16::from_bits(0x8000),
        ],
    );
    assert_invalid_argument(&f16_zeros_over_limit.validate(&f16_schema).unwrap_err());
}
