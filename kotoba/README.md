# Kotoba binding for Lance (honest v1)

This tree is a first-class sibling of `java/`, `python/`, and `rust/` **on this fork**. It is not a replacement for those implementations and is not proposed to [lance-format/lance](https://github.com/lance-format/lance).

## Scope

v1 parses only enough of a Lance **fragment data file** to identify a tiny vendored fixture:

- trailing 4-byte magic `LANC`
- the 16-byte v1 magics tail written by Lance (`i64` metadata position, `i16` major, `i16` minor, magic)

It does **not** implement a dataset reader, manifest/table format, page decoder, query engine, vector search, or any robotics/ML training path. It is **not robotics-ready**.

The compiler contract is Kotoba CLI **0.7.2**, `wasm32-kotoba-v1`, value profile `i64-v1`. No FFI. No IEEE floats.

## Fixture

`fixtures/tiny.lance` is a 93-byte historical v1 fragment copied from this repository's `test_data/v0.8.0/migrated_from_v0.7.5` data file. Last 16 bytes (little-endian):

```
41 00 00 00 00 00 00 00  00 00 01 00 4c 41 4e 43
meta_pos=65  major=0  minor=1  magic=LANC
```

`lance.kotoba` embeds those tail bytes as i64 scalars and returns:

```
meta_pos * 100000000 + magic_ok * 1000000 + major * 1000 + minor
= 6501000001
```

## Check

Requires Kotoba CLI 0.7.2 on `PATH` (or `$KOTOBA`). On linux-amd64, `checks.sh` also pins the published v0.7.2 binary SHA-256. Missing CLI, checksum mismatch, compile failure, or field mismatch is a failure. The script does not skip those as success.

```sh
# from this directory, with kotoba 0.7.2 on PATH
./checks.sh
```
