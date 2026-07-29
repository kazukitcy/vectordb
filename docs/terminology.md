# Terminology

Canonical vocabulary for names and documentation. List a term here only when a plausible synonym
was considered and rejected; record the rejected synonym so the debate is not rerun. Add, update,
or remove entries whenever a naming decision accepts or rejects a synonym.

| Term | Meaning | Do not use |
|---|---|---|
| entry (sparse vector) | One stored index/value pair; zero values are stored and counted | "non-zero": zeros are accepted, never dropped |
| document count | Unit of batch-write limits | "size": reads as bytes |
| vector dimension | Element count of a dense vector; index space (`0..dimension`) of a sparse vector | unqualified "dimension" in `Limits`: the cap is vector-only |
| field | A named, typed slot in a schema or document; `contains_field` tests name presence | bare "contains": reads as value containment |
| `pk` | The implicit string primary key; never a declarable field | "id": `DocId` is the internal u64, `pk` the user-facing key |
| `DocId` | Internal u64 identifier, allocated monotonically, never reused | conflating with `pk` |
| flush / durable commit | `flush()` completion is the only durability point; a successful write is not yet durable | "commit" for a mere WAL append |
| score | Smaller-is-better f32 kernel output (squared L2, negated dot); canonical in kernel API names | "distance" for negated-dot output: it is not a metric — reserve "distance" for prose about true metrics |
| kernel path | The instruction-set path a score kernel executes on (`KernelPath`) | "backend", "ISA level" |
| prefetch | Best-effort cache hint that a vector will be scored soon; may be a no-op | "preload": implies a guaranteed fetch |
| `l2_normalize` | In-place unit-L2 normalization returning the original norm as `f64` | "normalize_l2": breaks the `l2_*` prefix family's word order |
