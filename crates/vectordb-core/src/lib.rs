//! Shared data-model contracts for the vector database workspace.
//!
//! The shared data-model types of the vector database. Most users should import these types
//! through the `vectordb` facade instead.
//!
//! # Example
//!
//! ```
//! use vectordb_core::{DataType, Doc, FieldSchema, Schema};
//!
//! let schema = Schema::builder()
//!     .field(FieldSchema::builder("title", DataType::String).build()?)
//!     .build()?;
//! let mut doc = Doc::new();
//! doc.set_string("title", "A document");
//! doc.validate(&schema)?;
//! # Ok::<(), vectordb_core::Error>(())
//! ```

mod data_type;
mod doc;
mod error;
mod f16;
mod limits;
mod schema;

#[doc(inline)]
pub use data_type::{DataType, DocId, MetricType};
#[doc(inline)]
pub use doc::Doc;
#[doc(inline)]
pub use error::{Error, Result};
#[doc(inline)]
pub use f16::F16;
#[doc(inline)]
pub use limits::Limits;
#[doc(inline)]
pub use schema::{FieldSchema, FieldSchemaBuilder, Schema, SchemaBuilder};
