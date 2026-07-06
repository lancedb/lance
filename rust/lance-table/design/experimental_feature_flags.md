# Proposal: A General Experimental-Feature Mechanism

Status: **Proposed** (policy + format change; seeking PMC agreement)

## Summary

Add a single, general mechanism for shipping *experimental* format features
without spending a scarce, permanent feature-flag bit per experiment:

- Reserve **one** persisted feature-flag bit, `FLAG_EXPERIMENTAL` (`1 << 6`),
  meaning "this dataset uses one or more experimental features."
- Carry the actual feature identities in **free-form string lists** on the
  manifest: `experimental_reader_features` and `experimental_writer_features`.
- A build understands an experiment only if its name is in a compile-time
  registry (gated by the experiment's Cargo feature). Unknown name → refuse.

This lets experiments be **free-flowing** (mint/abandon names at will, unbounded
namespace) while the permanent bitmap stays **conservative** (a bit is spent
only when a feature graduates).

## Motivation

Feature flags today are two `u64` bitmaps (`reader_feature_flags`,
`writer_feature_flags`) with a linear "first unknown bit" boundary: an
implementation accepts a dataset iff no bit at or above the boundary is set.
This is elegant for *stable* features, but it makes experimentation expensive:

1. **Every persisted bit is burned forever.** Even an *abandoned* experiment's
   bit can never be reused, because some dataset may have written it and reusing
   the bit would misread that dataset. Experimental churn permanently pollutes a
   scarce (64-bit), shared namespace.
2. **Claiming a bit is high-ceremony**, which chills experimentation — reaching
   for a permanent, format-committing bit for a maybe-throwaway idea feels
   heavy, so people avoid trying things.

Only *format-changing* experiments need a persisted flag at all (purely
behavioral experiments are ordinary Cargo features / runtime config and never
touch the bitmap). But for the ones that do, we want a cheap, safe, churnable
path.

## Design

### The bit is the fail-closed anchor; the strings are the capability list

`FLAG_EXPERIMENTAL` reuses bit 64, which was previously the first `FLAG_UNKNOWN`
bit. This is safe to repurpose: until now, writing bit 64 made a dataset
unreadable by *every* implementation, so no dataset in the wild has it set.

- **Old implementations** (predating any experiment) still treat bit 64 as
  unknown and reject the dataset on the bit alone — they never need to parse the
  names. This is the whole reason the bit exists.
- **New implementations** ignore the bit for admission and instead check the
  string lists directly: a reader admits the dataset iff it understands every
  name in `experimental_reader_features` (and likewise for writes).

The bit is thus **backward-facing** (fail-closed for old readers) and the
strings are **forward-facing** (precise capability negotiation for new readers).

Why the bit is load-bearing and a string list alone is unsafe: a new manifest
field is, by protobuf's rules, silently ignored by readers that don't know it.
Without the bit, an old reader would skip the experimental-features field and
happily misread experimental data. The bit forces the rejection.

### Admission algorithm

```text
can_read(reader_flags, experimental_reader_features):
    if any bit >= FLAG_UNKNOWN (128) is set:      # a non-experimental unknown
        reject
    if any name in experimental_reader_features is not in this build's registry:
        reject
    accept
```

`can_write` is identical over the writer flags/list. `FLAG_EXPERIMENTAL` itself
is a *known* bit (below the new `FLAG_UNKNOWN = 128`), so it never trips the
first check; gating is entirely by the name lists.

### The registry

Each build exposes the experimental features it understands via a compile-time
list, populated only when the implementing Cargo feature is enabled:

```rust
pub fn known_experimental_features() -> &'static [&'static str] {
    &[
        #[cfg(feature = "unstable-action-transactions")]
        "action-transactions",
    ]
}
```

A default release build understands *none*, so it rejects any dataset that
declares an experimental feature — the desired conservative default.

### Two independent layers of gating

- **Cargo feature** (compile-time): keeps experimental *code* out of released
  artifacts and out of the support/attack surface.
- **`FLAG_EXPERIMENTAL` + name list** (persisted): keeps experimental *data*
  safe across versions and implementations.

They are orthogonal and both apply. The mechanism itself (bit semantics + name
checking) is **not** Cargo-gated — it must be in every build so that all builds
fail closed on experiments they lack.

## Graduation and lifecycle

- **Graduate:** when an experiment is ratified, it gets its own dedicated stable
  feature-flag bit. Its name is dropped from `known_experimental_features`, and
  new writes stop emitting the name (setting the new stable bit instead). Only
  now does the feature consume a permanent bitmap slot — and only because it
  earned one.
- **Abandon:** drop the name from the registry. Datasets that used it become
  unreadable by all builds (correctly — the feature is gone). No permanent bit
  was ever spent.
- **Debt control:** the registry is a single auditable list. An experimental
  name that has not graduated within N releases should be removed. Experimental
  code lives behind its Cargo feature in isolated modules, not threaded through
  stable paths.

Be **free-flowing in the string list, conservative at the bitmap.** The speed
bump lands at graduation (a real compatibility commitment), not at
experiment-start (where friction just deters trying things).

## Alternatives considered

- **A dedicated bit per experiment** (what the action-transactions skeleton PR
  #7641 currently does). Simple, but every experiment permanently burns a scarce
  bit even if abandoned. Fine for one feature; doesn't scale to a culture of
  experimentation.
- **A free-form field with no bit.** Unsafe: old readers silently ignore the
  unknown field and misread experimental data. The reserved bit is required as
  the fail-closed anchor.
- **A long-lived feature branch instead of flags.** Different tool — it isolates
  *code*, not *runtime/format behavior on a shared codebase*. It defers
  integration (merge rot, big-bang review), blocks early opt-in dogfooding, and
  *still* needs a persisted marker so branch-written data is identifiable and
  rejectable. Branches remain the right tool for short throwaway spikes and
  compose fine as the per-slice PR mechanism, but not as the isolation model for
  a feature.

## Prior art

Delta Lake's "table features" made exactly this move: replacing a coarse,
monotonic protocol-version integer with a named, split reader/writer feature set
so that features could be added independently and negotiated by name. This
proposal is the same shape, anchored to Lance's existing feature-flag bitmap for
backward-compatible rejection.

## Format change

Two `repeated string` fields on `Manifest` (`table.proto`):
`experimental_reader_features = 22`, `experimental_writer_features = 23`, plus
documentation of bit `1 << 6` on `reader_feature_flags`. Backwards compatible
(new field numbers, new bit that old readers already reject).

## Impact on #7641 (action-based transactions)

The action-transactions skeleton (#7641) currently claims a dedicated
`FLAG_ACTION_TRANSACTIONS` bit. Once this mechanism lands, #7641 should adopt it
instead: set `FLAG_EXPERIMENTAL` + list `"action-transactions"`, register that
name under the `unstable-action-transactions` Cargo feature, and free the
dedicated bit. This PR should land first.

## Open questions

- Naming convention for feature identifiers (proposed: lowercase-kebab, stable
  across the experiment's life).
- Whether to also expose the registry / active experimental features through the
  Python and Java bindings for introspection.
- Whether graduation should keep the old experimental name readable (dual-accept
  during a transition window) or require a rewrite.
