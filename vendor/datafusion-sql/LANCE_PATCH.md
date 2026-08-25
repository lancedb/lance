# Lance DataFusion SQL patch

This directory contains the `datafusion-sql` 54.1.0 crate source with the
unparser stack-overflow fix from
[apache/datafusion#23058](https://github.com/apache/datafusion/pull/23058) and
the race-free local stack-growth follow-up from
[apache/datafusion#23284](https://github.com/apache/datafusion/pull/23284)
backported.

DataFusion 55.0.0 contains both fixes, but it requires Rust 1.94 and Arrow 59.2.
Lance currently supports Rust 1.91 and uses Arrow 58, so it cannot adopt that
release yet. Remove this vendored crate and the three Cargo patch entries after
Lance can upgrade to a compatible DataFusion release containing both fixes.

The root, Python, and Java source builds use their local `[patch.crates-io]`
entries and enable `recursive_protection`. Cargo does not include a workspace
patch in published crate manifests, so downstream users of published Lance Rust
crates still resolve the registry `datafusion-sql` until the DataFusion upgrade.
