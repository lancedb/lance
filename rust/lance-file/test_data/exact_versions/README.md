# Exact File-Version Compatibility Fixtures

These files were generated with the writers at baseline commit
`3a72f8a61e14613f517dded6816d4bfc77817c93`. The deterministic input batch is
defined by `compatibility_fixture_batch` in `src/writer_tests.rs` and covers
primitive, nullable UTF-8, nullable list, nullable dictionary, blob, multiple
input batches, and multiple pages.

The baseline generator was run twice in separate processes with a Cargo target
directory isolated from the refactored checkout. Both runs produced identical
bytes:

| File | SHA-256 |
| --- | --- |
| `v1.lance` | `fa8b3d81b9d4fd4ade5a7c3d077ebf2155664e12b9335e26fac1c0d0774e916c` |
| `v2_0.lance` | `073c8c24eb4433b83d0dda95bf7a731a9f5d8f32d78440f2f391474e99b9c49a` |
| `v2_1.lance` | `3af97ba176b72c7e00a248b4a270a53402a72e594631950f76eb3daab45c50ce` |
| `v2_2.lance` | `8298cd9301e657417b0725461345c27cf46515529d2a8b35824be139e3466a14` |

The compatibility tests require each refactored stable writer to reproduce its
fixture byte-for-byte and each refactored reader to open and read the baseline
file. V2.3 is unstable, so it has deterministic current-revision tests instead
of a checked-in compatibility fixture.

Regenerate these fixtures only from the baseline writer APIs. Files generated
with the implementation under test are not independent compatibility evidence.
