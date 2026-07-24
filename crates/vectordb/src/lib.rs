//! Public API facade for the embedded vector database.
//!
//! One stable import path for the vector database's public types.

#[doc(inline)]
pub use vectordb_core::{
    DataType, Doc, DocId, Error, F16, FieldSchema, FieldSchemaBuilder, Limits, MetricType, Result,
    Schema, SchemaBuilder,
};
