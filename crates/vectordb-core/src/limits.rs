// All limits live in one struct, rather than as per-module constants, so a collection's whole
// operating envelope is captured by the Schema that validated it. Later milestones enforce the
// reserved limits from this same struct instead of introducing separate knobs.
// Fields are deliberately public: Limits carries no internal invariant (any combination is a
// valid input), so getters would only add ceremony. Validation happens where limits are consumed.
/// Configurable resource and request limits.
///
/// Applications may start with [`Limits::default`], adjust individual limits, and then build a
/// [`Schema`](crate::Schema).
#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Limits {
    // For dense vectors this caps the stored element count; for sparse vectors it caps the index
    // space, which is independently bounded by the u32 coordinate type (see schema.rs).
    /// Upper bound on the dimension of a dense or sparse vector field.
    ///
    /// Dimension validation applies further rules beyond this bound; they are documented on
    /// [`FieldSchemaBuilder::build_with_limits`](crate::FieldSchemaBuilder::build_with_limits).
    pub max_vector_dimension: usize,
    // Deliberately counts every stored entry. Validation does not classify or drop zero-valued
    // entries, so the limit applies to stored entries exactly as supplied.
    /// Maximum number of stored entries in one sparse vector.
    ///
    /// Entries containing `0.0` or `-0.0` are accepted and count toward this limit; validation
    /// does not remove them implicitly.
    pub max_sparse_vector_entries: usize,
    // Aggregate index-building, memory, and flush costs grow roughly linearly with the number of
    // vector fields, so this cap prevents runaway schemas. Scalar and array field counts are
    // deliberately unlimited.
    /// Maximum number of vector fields in one collection schema.
    pub max_vector_fields: usize,
    // Reserved: enforced by SearchRequest validation from M2 onward. Defined now so the limit
    // vocabulary stays in one place from the start.
    /// Maximum `top_k` value accepted by search requests.
    pub max_top_k: usize,
    // Reserved: enforced by the batch write API from M3 onward. This limits the work performed in
    // one WAL critical section. Batches are not atomic, so the limit controls resource use rather
    // than guaranteeing atomicity.
    /// Maximum number of documents accepted by one batch write.
    pub max_batch_write_documents: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_vector_dimension: 16_384,
            max_sparse_vector_entries: 65_536,
            max_vector_fields: 4,
            max_top_k: 65_536,
            max_batch_write_documents: 10_000,
        }
    }
}
