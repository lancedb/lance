# Independent producer fixture

These are exact OCI metadata and Arrow schema bytes from an image built with
Docker/BuildKit and exported by Skopeo on 2026-09-07. The image contains CPython
3.12.14, PyArrow 21.0.0, NumPy 2.3.2, and an image-owned scalar function. Its
deliberately unusable Entrypoint/Cmd are retained to test that metadata parsing
does not invoke the image or rewrite its runtime configuration.

- Manifest: `sha256:7e22f815b6648e14f093a3979a8e5a2082fa773ebe1ec84b135cae7e84d6f8e6`
- Config: `sha256:b2365936ab77fbbbc896ddb765a4724facf60b99ac077a5027833a8880861494`

The schema messages were produced inside that image by
`schema.serialize().to_pybytes()` and copied out unchanged. They were not written
by the Rust parser or its tests. Definitions:

- Input: nullable `x: float64`, schema metadata `fixture=scalar-v1`.
- Output: nullable `value: float64`.
- Initialization: non-null `factor: float64`, non-null `mode: utf8`.
- Empty initialization: zero fields.
- Nested: nullable struct `nested` with non-null `vector: fixed_size_list<float32>[3]`
  and nullable `labels: list<utf8>`, schema metadata `fixture=nested-vector`.
  List item fields are named `item` and are nullable.

The files preserve metadata from the independently built image for parser and
interoperability tests. Filesystem layers and executable code are intentionally
not included; this directory is not a complete, runnable OCI image layout.
