use vectordb::{DataType, Doc, DocId, Error, F16, FieldSchema, Limits, MetricType, Result, Schema};

#[test]
fn facade_reexports_the_public_core_contract() -> Result<()> {
    let field = FieldSchema::builder("embedding", DataType::DenseVectorF16)
        .dimension(1)
        .build()?;
    let schema = Schema::builder().field(field).build()?;
    let mut doc = Doc::new();
    doc.set_dense_vector_f16("embedding", [F16::from_f32(1.0)]);
    doc.validate(&schema)?;

    let _: MetricType = MetricType::Cosine;
    let _: DocId = DocId::new(1);
    let _: Limits = Limits::default();
    let _: Option<Error> = None;
    Ok(())
}
