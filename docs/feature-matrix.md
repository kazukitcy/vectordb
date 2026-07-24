# Feature Matrix

This document translates the planned capabilities in `todo.md` §2 and the deliberate exclusions in §4.
It describes roadmap commitments, not functionality already implemented. Milestone labels identify when
the capability is intended to become operational.

## Field and Index Compatibility

`—` means the combination is not planned for v1. Vector metric and quantization columns do not apply to
scalar or text indexes.

| Field type | Flat | HNSW | IVF | Streaming graph | Disk graph | Scalar inverted | FTS | Metrics | Quantization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Bool`, integer, or float scalar | — | — | — | — | — | M5 | — | — | — |
| Scalar array | — | — | — | — | — | M5 | — | — | — |
| `String` | — | — | — | — | — | M5 | M7 | — | — |
| `Binary` | — | — | — | — | — | — | — | — | — |
| Dense `F32` | M2 | M4 | M6 | M11 | M11 | — | — | L2, inner product, cosine | F16, I8, I4 (M6); PQ for disk graph (M11) |
| Dense `F16` | M2 | M4 | M6 | M11 | M11 | — | — | L2, inner product, cosine | F16, I8, I4 (M6); PQ for disk graph (M11) |
| Dense `I8` | M2 | M4 | M6 | M11 | M11 | — | — | L2, inner product, cosine | F16, I8, I4 (M6); PQ for disk graph (M11) |
| Sparse `F32` or `F16` | M9 | M9 | — | — | — | — | — | Inner product only | — |

Scalar coverage includes `Bool`, `I32`, `I64`, `U32`, `U64`, `F32`, and `F64`, plus arrays of each
scalar type. M6 must define which dense source types can be quantized into which index formats; the
roadmap currently commits to F16/I8/I4 quantized indexes but does not define the full compatibility
matrix between source types and quantized formats. I4 requires an even vector dimension. An optional
learned rotation may precede I8 or I4.

Flat is the fallback when a vector field has no explicitly built index. Sparse HNSW shares the dense
graph structure but only replaces storage and distance evaluation. PQ is scoped to the M11 disk-resident
graph, where in-memory PQ codes guide traversal and raw vectors provide exact distances for visited nodes.

## Query Capability Milestones

| Query capability | Contract introduced | Operational support |
| --- | --- | --- |
| Radius threshold | M2 `SearchRequest` | M2 at index level; M8 through the integrated planner |
| Forced linear/exact search | M2 `SearchRequest.exact` | M2 at index level; M8 through the integrated planner |
| Refine/rerank | M2 `SearchRequest.refine` | M6 with quantized candidates and raw-vector reranking |
| Group-by | M2 collector extension point | M8 collector, with extended exploration while groups remain underfilled |
| Hybrid search | — | M8 for multiple vector subqueries plus FTS, with RRF, weighted, or callback fusion |

When `exact` is enabled, refine is disabled. Radius combined with top-k means “take at most top-k results
from those within the threshold.” Hybrid per-query candidate limits are independent of final top-k.

## Deliberate Non-Support for v1

The following list is synchronized with `todo.md` §4:

- Distributed operation, replication, and server mode; the database is in-process only.
- Atomic multi-document transactions; batch operations report status per document.
- Multiple concurrent writer processes; the design permits one writer and multiple readers.
- Full SQL, including `SELECT`; only a WHERE-like filter expression is planned.
- Binary vectors and F64 vectors. Their enum variants may be reserved for future compatibility.
- Backup, or export of a consistent snapshot. The M3 layout only preserves room for a future
  implementation.
- Guaranteed operation on NFS or other network filesystems.

MIPS-to-L2 conversion in M6 is optional. If it is not implemented, it must be moved into this deliberate
non-support list before M6 is considered complete.
