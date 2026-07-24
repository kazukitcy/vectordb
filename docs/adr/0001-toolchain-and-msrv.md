# ADR 0001: Toolchain, MSRV, and Clippy Policy

- Status: Accepted
- Date: 2026-07-21

## Context

The workspace needs reproducible compiler behavior, an explicit minimum supported Rust version (MSRV),
and a lint baseline that catches API and correctness problems without making low-value style warnings
fatal by default.

## Decision

Pin the stable compiler available at project inception, Rust 1.93.1, in `rust-toolchain.toml`. Set the
workspace `rust-version` and MSRV to the same version. Before 1.0, a minor release may raise the MSRV;
the release notes must call out the change and this ADR or a superseding ADR must record the rationale.

Enable Clippy's `pedantic` group at warning level. Selectively allow noisy lints in the root
`[workspace.lints.clippy]` table. Each allow entry must have an adjacent comment explaining why the lint
does not fit this codebase. CI promotes all remaining warnings to errors with `-D warnings`; source code
does not use a crate-level `#![deny(warnings)]` attribute.

## Consequences

- Local and CI builds use the same stable compiler, rustfmt, and Clippy behavior.
- Contributors on older Rust releases must upgrade to the recorded MSRV.
- MSRV changes before 1.0 are permitted but visible and tied to minor releases.
- The lint policy remains strict while exceptions stay reviewable at one workspace-level location.

## Alternatives Considered

- Tracking the moving `stable` channel was rejected because it makes formatting and lint output change
  without a repository change.
- Enabling every pedantic lint without exceptions was rejected because recurring low-signal warnings
  would obscure correctness and API findings.
- Applying `-D warnings` in source was rejected because it makes downstream compilation sensitive to
  compiler lint changes outside CI policy.

## References

- `todo.md` §0, M0, and §3
- `rust-toolchain.toml`
- Root `[workspace.lints]` in `Cargo.toml`
