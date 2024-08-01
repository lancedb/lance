# lance-encoding-datafusion

`lance-encoding-datafusion` is an internal sub-crate, containing encoders and
decoders for the Lance file format that rely on Datafusion. Partly this is to
keep the size of `lance-encoding` small and partly this is to prove that
encodings are extensible.

**Important Note**: This crate is **not intended for external usage**.
