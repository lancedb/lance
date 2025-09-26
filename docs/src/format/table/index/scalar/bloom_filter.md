# Bloom Filter Index

Bloom filters are probabilistic data structures that allow for fast membership testing.
They are space-efficient and can test whether an element is a member of a set.
It's an inexact filter - they may include false positives but never false negatives.

## Index Details

```protobuf
%%% proto.message.BloomFilterIndexDetails %%%
```

## Storage Layout

The bloom filter index stores zone-based bloom filters in a single file:

1. `bloomfilter.lance` - Bloom filter statistics and data for each zone

### Bloom Filter File Schema

| Column              | Type    | Nullable | Description                                     |
|---------------------|---------|----------|-------------------------------------------------|
| `fragment_id`       | UInt64  | false    | Fragment containing this zone                   |
| `zone_start`        | UInt64  | false    | Starting row offset within the fragment         |
| `zone_length`       | UInt64  | false    | Number of rows in this zone                     |
| `has_null`          | Boolean | false    | Whether this zone contains any null values      |
| `bloom_filter_data` | Binary  | false    | Serialized SBBF (Split Block Bloom Filter) data |

### Schema Metadata

| Key                       | Type   | Description                                                 |
|---------------------------|--------|-------------------------------------------------------------|
| `bloomfilter_item`        | String | Expected number of items per zone (default: "8192")         |
| `bloomfilter_probability` | String | False positive probability (default: "0.00057", ~1 in 1754) |

## Bloom Filter Spec

The bloom filter index uses a Split Block Bloom Filter (SBBF) implementation,
which is optimized for SIMD operations.

### SBBF Structure

The SBBF divides the bit array into blocks of 256 bits, where each block consists of 8 contiguous 32-bit words.
This structure enables efficient SIMD operations and cache-friendly memory access patterns.
The block layout is the following:

- **Block size**: 256 bits (32 bytes)
- **Words per block**: 8 × 32-bit integers
- **Minimum filter size**: 32 bytes (1 block)
- **Maximum filter size**: 128 MiB

### Hashing Mechanism

The SBBF uses xxHash64 with seed=0 for primary hashing, combined with a salt-based secondary hashing scheme:

1. **Primary hash**: xxHash64(value) → 64-bit hash
2. **Block selection**: Upper 32 bits determine which block to use
3. **Bit selection**: Lower 32 bits combined with 8 salt values set 8 bits in the block

#### Salt Values

```
SALT = [0x47b6137b, 0x44974d91, 0x8824ad5b, 0xa2b7289d,
        0x705495c7, 0x2df1424b, 0x9efc4947, 0x5c6bfb31]
```

Each salt value generates one bit position within the block, ensuring uniform distribution.

### Filter Sizing Algorithm

The SBBF automatically determines optimal filter size based on:
- **NDV** (Number of Distinct Values): Expected unique items
- **FPP** (False Positive Probability): Target error rate

The implementation uses binary search to find the minimum log₂(bytes) that achieves the desired FPP,
using Putze et al.'s cache-efficient bloom filter formula.

### False Positive Probability Convergence

The implementation uses up to 750 iterations of Poisson distribution calculations to ensure accurate FPP estimation,
particularly for dense filters where NDV approaches filter capacity.

## Accelerated Queries

The bloom filter index provides inexact results for the following query types:

| Query Type | Description               | Operation                                 | Result Type |
|------------|---------------------------|-------------------------------------------|-------------|
| **Equals** | `column = value`          | Tests if value exists in bloom filter     | AtMost      |
| **IsIn**   | `column IN (v1, v2, ...)` | Tests if any value exists in bloom filter | AtMost      |
| **IsNull** | `column IS NULL`          | Returns zones where has_null is true      | AtMost      |