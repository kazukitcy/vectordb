use std::collections::HashSet;

use crate::{DataType, Doc, Error, Limits, Result};

pub(crate) const PRIMARY_KEY_FIELD_NAME: &str = "pk";

// Fields are private so that possessing a FieldSchema proves it passed builder validation.
// Code holding a FieldSchema can treat it as structurally valid; SchemaBuilder still re-runs
// the full validation against its own Limits, since a field may have been built under a
// different limit set. Contrast with Limits, whose fields are public because any value
// combination is valid input.
/// The validated definition of one document field.
///
/// Field names use an ASCII identifier form: the first byte is an ASCII letter or `_`, and remaining
/// bytes are ASCII alphanumeric characters or `_`. The exact name `pk` is reserved for the implicit
/// string primary key and cannot be declared in a schema.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FieldSchema {
    name: String,
    data_type: DataType,
    nullable: bool,
    dimension: Option<usize>,
}

impl FieldSchema {
    /// Starts a builder for a field with the given name and logical type.
    pub fn builder(name: impl Into<String>, data_type: DataType) -> FieldSchemaBuilder {
        FieldSchemaBuilder {
            name: name.into(),
            data_type,
            nullable: false,
            dimension: None,
        }
    }

    /// Returns the field name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the field's logical type.
    pub const fn data_type(&self) -> DataType {
        self.data_type
    }

    /// Returns whether the field accepts explicit null and may be omitted.
    pub const fn is_nullable(&self) -> bool {
        self.nullable
    }

    /// Returns the fixed vector dimension, or `None` for non-vector fields.
    pub const fn dimension(&self) -> Option<usize> {
        self.dimension
    }

    fn validate(&self, limits: &Limits) -> Result<()> {
        validate_field_name(&self.name)?;
        let max_sparse_dimension = u128::from(u32::MAX) + 1;

        match (self.data_type.requires_dimension(), self.dimension) {
            (true, None) => Err(Error::invalid_argument(format!(
                "vector field `{}` requires a dimension",
                self.name
            ))),
            (false, Some(_)) => Err(Error::invalid_argument(format!(
                "non-vector field `{}` cannot have a dimension",
                self.name
            ))),
            (true, Some(0)) => Err(Error::invalid_argument(format!(
                "vector field `{}` must have a non-zero dimension",
                self.name
            ))),
            (true, Some(dimension))
                if self.data_type.is_sparse_vector()
                    && dimension as u128 > max_sparse_dimension =>
            {
                Err(Error::invalid_argument(format!(
                    "sparse vector field `{}` dimension {dimension} exceeds the u32 coordinate capacity of {max_sparse_dimension}",
                    self.name
                )))
            }
            (true, Some(dimension)) if dimension > limits.max_vector_dimension => {
                Err(Error::invalid_argument(format!(
                    "vector field `{}` dimension {dimension} exceeds limit {}",
                    self.name, limits.max_vector_dimension
                )))
            }
            _ => Ok(()),
        }
    }
}

/// A chainable builder for [`FieldSchema`].
///
/// Validation is deferred until [`FieldSchemaBuilder::build`] or
/// [`FieldSchemaBuilder::build_with_limits`].
#[derive(Clone, Debug)]
#[must_use = "a field schema builder has no effect until it is built"]
pub struct FieldSchemaBuilder {
    name: String,
    data_type: DataType,
    nullable: bool,
    dimension: Option<usize>,
}

impl FieldSchemaBuilder {
    /// Sets whether this field accepts explicit null and may be omitted.
    pub const fn nullable(mut self, nullable: bool) -> Self {
        self.nullable = nullable;
        self
    }

    /// Sets the fixed dimension required by a vector field.
    pub const fn dimension(mut self, dimension: usize) -> Self {
        self.dimension = Some(dimension);
        self
    }

    /// Validates and builds the field using [`Limits::default`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] for an invalid or reserved name; a missing, unexpected,
    /// or zero dimension; a dimension above the default limit; or a sparse-vector dimension greater
    /// than the number of distinct `u32` coordinates (`u32::MAX + 1`).
    pub fn build(self) -> Result<FieldSchema> {
        self.build_with_limits(&Limits::default())
    }

    /// Validates and builds the field using the supplied limits.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] for an invalid or reserved name; a missing, unexpected,
    /// or zero dimension; a dimension above `limits.max_vector_dimension`; or a sparse-vector dimension
    /// greater than the number of distinct `u32` coordinates (`u32::MAX + 1`).
    pub fn build_with_limits(self, limits: &Limits) -> Result<FieldSchema> {
        let field = FieldSchema {
            name: self.name,
            data_type: self.data_type,
            nullable: self.nullable,
            dimension: self.dimension,
        };
        field.validate(limits)?;
        Ok(field)
    }
}

// Same invariant pattern as FieldSchema: private fields mean an existing Schema has already
// been checked for name uniqueness, the vector-field cap, and per-field validity under its
// own limits.
/// A validated collection schema and its configured limits.
///
/// The primary key is not represented here: every collection has an implicit string primary key named
/// `pk`, so declaring a field with that name is rejected by [`FieldSchemaBuilder`].
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Schema {
    fields: Vec<FieldSchema>,
    limits: Limits,
}

impl Schema {
    /// Starts an empty schema builder with [`Limits::default`].
    pub fn builder() -> SchemaBuilder {
        SchemaBuilder::default()
    }

    /// Returns all explicit fields in insertion order.
    pub fn fields(&self) -> &[FieldSchema] {
        &self.fields
    }

    /// Looks up an explicit field by its exact name.
    pub fn field(&self, name: &str) -> Option<&FieldSchema> {
        self.fields.iter().find(|field| field.name == name)
    }

    /// Returns the limits captured when this schema was built.
    pub const fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Validates a document against this schema.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if the document contains an unknown field; omits or stores
    /// null in a non-nullable field; stores a value with the wrong logical type or vector shape;
    /// contains a value that exceeds a configured limit; or contains a non-finite floating-point value.
    pub fn validate_doc(&self, doc: &Doc) -> Result<()> {
        doc.validate(self)
    }
}

/// A chainable builder for [`Schema`].
///
/// Fields may be added in any order. Cross-field validation, including duplicate names and the vector
/// field count, occurs in [`SchemaBuilder::build`].
#[derive(Clone, Debug, Default)]
#[must_use = "a schema builder has no effect until it is built"]
pub struct SchemaBuilder {
    fields: Vec<FieldSchema>,
    limits: Limits,
}

impl SchemaBuilder {
    /// Adds an explicit field definition.
    pub fn field(mut self, field: FieldSchema) -> Self {
        self.fields.push(field);
        self
    }

    /// Replaces the configurable limits used to validate the schema. The resulting [`Schema`]
    /// retains these limits.
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Validates and builds the schema.
    ///
    /// Every field is revalidated against the schema's limits, even if it was originally built with a
    /// different limit set.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] when a field is invalid under the selected limits, field names
    /// are duplicated, or the number of vector fields exceeds `limits.max_vector_fields`.
    pub fn build(self) -> Result<Schema> {
        let mut names = HashSet::with_capacity(self.fields.len());
        let mut vector_field_count = 0;

        for field in &self.fields {
            field.validate(&self.limits)?;
            if !names.insert(field.name.as_str()) {
                return Err(Error::invalid_argument(format!(
                    "duplicate field name `{}`",
                    field.name
                )));
            }
            if field.data_type.is_vector() {
                vector_field_count += 1;
            }
        }

        if vector_field_count > self.limits.max_vector_fields {
            return Err(Error::invalid_argument(format!(
                "schema has {vector_field_count} vector fields, exceeding limit {}",
                self.limits.max_vector_fields
            )));
        }

        Ok(Schema {
            fields: self.fields,
            limits: self.limits,
        })
    }
}

fn validate_field_name(name: &str) -> Result<()> {
    if name == PRIMARY_KEY_FIELD_NAME {
        return Err(Error::invalid_argument(
            "field name `pk` is reserved for the implicit string primary key",
        ));
    }

    let mut bytes = name.bytes();
    let Some(first) = bytes.next() else {
        return Err(Error::invalid_argument("field name cannot be empty"));
    };
    if !(first.is_ascii_alphabetic() || first == b'_')
        || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(Error::invalid_argument(format!(
            "field name `{name}` must be an ASCII identifier"
        )));
    }

    Ok(())
}
