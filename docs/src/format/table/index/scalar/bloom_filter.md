# Bloom Filter Index

Bloom filters are probabilistic data structures that test set membership with possible false positives but no false negatives.

## Index Details

```protobuf
%%% proto.message.BloomFilterIndexDetails %%%
```

## Storage Layout

The bloom filter index is currently integrated with other index types and doesn't have a standalone implementation in Lance. It's typically used as an auxiliary structure within other indices.

## Implementation Details

- **Hash Functions**: Multiple hash functions to set/check bits
- **False Positive Rate**: Configurable based on expected elements
- **Space Efficiency**: Compact bit array representation
- **No Deletions**: Standard bloom filters don't support removal