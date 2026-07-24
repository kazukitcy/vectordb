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
