# Repository Guide

## Commands

Run all commands from the repository root.

- Build: `cargo build --workspace`
- Format: `cargo fmt --all --check`
- Lint: `cargo clippy --workspace --all-targets -- -D warnings`
- Test: `cargo test --workspace`
- Dependency policy: `cargo deny check`

Use test-driven development: explore the relevant contract, add or identify a failing test, make the
smallest change that passes, and then refactor while the checks remain green.

## Workspace

- `vectordb` is the public facade and assembles the other workspace crates.
- `vectordb-core` owns shared API contracts, types, configuration, limits, and errors.
- `vectordb-simd` owns scalar reference and runtime-dispatched vector score kernels.
- `vectordb-index` owns vector indexes and quantization.
- `vectordb-store` owns segments, durability, recovery, and read views.
- `vectordb-scalar` owns scalar inverted indexes and filter execution.
- `vectordb-text` owns tokenization, full-text indexes, and BM25 search.

Feature crates depend on `vectordb-core`; the facade composes all feature crates. Keep shared dependency
versions in the root `[workspace.dependencies]` table and lint policy in `[workspace.lints]`.

## Coding Conventions

- Separate state ownership from logic, and keep public contracts strict enough that implementations can
  be replaced or regenerated behind them.
- Limit unsafe code to exactly three audited boundaries: SIMD kernels, mmap boundaries, and the graph
  adjacency-list publication protocol. Do not introduce unsafe code elsewhere.
- Convert dependency errors into the repository's own `Error` type at crate boundaries.
- Never expose types from dependency crates in public API signatures.
- Rustdoc documents the public contract only: behavior, guarantees, and error conditions. Keep
  implementation details and design rationale (why, and why not) out of rustdoc; put them in regular
  `//` comments next to the code they govern, or in an ADR.
- Put statically checkable rules in rustc/clippy configuration or a dedicated lint rather than prose.
- Keep public documentation and commit messages in English.

## Terminology

`docs/terminology.md` is the canonical vocabulary. Consult it before naming public items or
writing documentation, and add, update, or remove entries whenever a naming decision accepts or
rejects a synonym.

## Architecture Decision Records

For a decision that is expensive to reverse:

1. Copy `docs/adr/0000-template.md` to a file named with the next zero-padded sequence number and a
   short kebab-case title.
2. Fill in the context, decision, consequences, and considered alternatives before implementation.
3. Set the status to `Proposed`; change it to `Accepted` when the decision is approved.
4. Link the ADR from relevant code or documentation. Superseded records remain in place and point to the
   replacement ADR.

## Roadmap and Completion

`todo.md` is the single source of truth for the roadmap. Work through milestones in order and judge
progress by each milestone's explicit completion conditions. Do not mark a checkbox merely because code
exists; all associated tests, documentation, and CI gates must satisfy the completion condition.

## Required GitHub Checks

Required checks are repository settings and cannot be enabled by this codebase. After the workflow has
run at least once, a repository administrator must open **Settings → Rules → Rulesets** (or branch
protection for `main`), require pull requests and status checks, and select:

- `stable (Linux x86_64)`
- `stable (macOS aarch64)`
- `sanitizer (ASan)`
- `sanitizer (TSan)`
- `cargo-deny`

Do not require `miri (codec allowlist placeholder)` yet. The job is a policy placeholder that always
passes until the first pure codec tests are added. Once those tests exist, replace the placeholder
with actual `cargo +nightly miri test` commands and add the job to the required list.

Keep this list synchronized with `.github/workflows/ci.yml` whenever job names change.
