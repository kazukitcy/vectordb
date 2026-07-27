# ADR 0002: Score Orientation and Kernel Scope

- Status: Accepted
- Date: 2026-07-25

## Context

The score kernels in `vectordb-simd` feed every vector index that later
milestones build (Flat, HNSW, IVF, graph). Search collects candidates through
heap-based top-k merging, and every consumer must agree on which direction is
"better" for every metric. The kernels are also the first of the three audited
unsafe boundaries, so their numeric contracts (accumulation width, overflow
bounds, exactness claims) must be fixed before SIMD implementations exist, and
the equivalence tests that police those implementations need an oracle that can
actually detect lane-level bugs.

Two toolchain facts constrain the M1 scope (verified by compile probes on the
pinned Rust 1.93.1): the AArch64 dot-product intrinsics (`vdotq_s32`, feature
gate `stdarch_neon_dotprod`) and the fp16 NEON intrinsic types including the
`vcvt_f32_f16` family (`stdarch_neon_f16`) are unstable and unusable on stable.

## Decision

**Score orientation.** Every kernel returns an `f32` score where smaller is
better:

| Metric | Kernel output |
| --- | --- |
| `L2` | squared Euclidean distance `Σ (a_i − b_i)²` (no square root) |
| `InnerProduct` | negated dot product `−Σ a_i·b_i` |
| `Cosine` | negated dot product over pre-normalized inputs |

The public surface names this output `score` (`ScoreKernel`, `score`,
`score_many`, `score_contiguous`); "distance" is reserved for prose about true
metrics. Kernels negate the accumulated dot once, after the final reduction, so
exact-cancellation inputs produce the same signed zero on every path.

**Cosine.** Cosine is insert-time L2 normalization plus the negated-dot kernel.
`vectordb-simd` ships the normalization helpers (`l2_norm`, `normalize_l2`);
performing normalization at insert time and retaining the original norm is the
storage layer's obligation (M3). The helpers return the norm as `f64` because a
valid f32 vector's norm can exceed `f32::MAX`, and division uses the f64 value.

**i8 kernels.** i8 kernels accumulate in `i32` and convert to `f32` once at the
end; scalar and SIMD paths must agree bit-exactly on the resulting `f32`.
Squared L2 is exact through dimension 33,025 and overflows `i32` beginning at
dimension 33,026 (`33_026 · 65_025 = 2_147_515_650 > i32::MAX`), so the kernels
enforce `MAX_I8_DIMENSION = 32_768` as a panic precondition. "Bit-exact" is
scoped to equality after the final conversion: above 2²⁴ the conversion itself
rounds, and distinct integer scores can collide; this resolution loss is
accepted. The i8 `Cosine` metric maps to the same negated-dot kernel; it
assumes inputs quantized from unit-normalized f32 vectors, and the end-to-end
i8 cosine semantics (scale/bias metadata and distance correction terms) are
defined in M6. AVX-512 availability for i8 is keyed on the metric: squared L2
requires AVX-512F+BW only, while the dot metrics additionally require
AVX-512VNNI (the dot kernel executes `vpdpbusd`; the L2 kernel widens with BW
instructions).

**Numeric envelope.** Float kernels accumulate in `f32`. Inputs whose
intermediate sums exceed the f32 range can produce non-finite scores (±inf, or
NaN from `inf − inf` cancellation) even though every component is finite. This
envelope is a documented part of the kernel contract; how non-finite scores are
ordered is decided by the M2 collector contract.

**F16 layout.** `vectordb_core::F16` is `#[repr(transparent)]` over its inner
half-precision value and is layout-identical to the IEEE 754 binary16 bit
representation `u16`. This is a public layout guarantee: SIMD kernels load
`&[F16]` memory directly as u16 lanes.

## Consequences

- One score orientation removes per-metric branching from collectors and
  merges; scores are not user-facing similarities, and M8 defines any
  presentation mapping.
- The F16 layout guarantee constrains future changes: the internal
  representation must remain a transparent binary16 value, and upgrades of the
  backing dependency must preserve its transparent-u16 layout.
- NEON i8 uses the widening multiply-add sequence on every AArch64 CPU, and
  NEON f16 converts to f32 through fixed-size stack chunks; the dot-product and
  native-fp16 instructions are deferred until their intrinsics stabilize.
  AVX-512 execution in CI is conditional on runner hardware; path-selection
  correctness is covered by synthetic-mask resolver tests, and the AVX-512 VNNI
  signedness correction is verified by a portable emulation test that runs on
  every CI runner.
- Schema-level enforcement of `MAX_I8_DIMENSION` (rejecting i8 vector fields
  above the kernel cap at schema/insert validation) is an M3 follow-up; until
  then the kernel panic is the only guard.
- Non-finite score ordering is deliberately unspecified in M1 and must be
  settled by the M2 collector contract.

## Alternatives Considered

- **Per-metric orientation flags** (each consumer branches on
  distance-vs-similarity): rejected — pushes a correctness-critical branch into
  every collector and merge site.
- **Runtime normalization inside the cosine kernel**: rejected — doubles the
  hot-path cost and hides the storage-layer norm contract.
- **Square-rooted L2**: rejected — the square root is monotone, adds cost, and
  loses exactness in the integer-grid equivalence oracle.
- **Inline `asm!` for the AArch64 `sdot` instruction**: rejected for M1 —
  hand-written encodings bypass the compiler's feature tracking and add an
  audit burden the widening sequence avoids.
- **i64 accumulation for i8 kernels**: rejected — halves SIMD throughput to
  lift a cap far above the default vector-dimension limit.

## References

- Roadmap: `todo.md` §0 (design principles), M1, M2 (collector contract), M3
  (storage normalization, schema cap), M6 (quantization correction terms).
- ADR 0001 (toolchain pin: stable 1.93.1, the source of the intrinsic
  stability constraints).
- Rust tracking issues for the deferred intrinsics: rust-lang/rust#117224
  (`stdarch_neon_dotprod`), rust-lang/rust#136306 (`stdarch_neon_f16`).
