# N-gram Index

N-gram indices break text into overlapping sequences for substring matching and fuzzy search.

## Index Details

```protobuf
%%% proto.message.NGramIndexDetails %%%
```

## Storage Layout

The N-gram index is currently not implemented as a standalone index in Lance. The concept is used within the inverted index for tokenization.

## Implementation Details

- **N-gram Generation**: Configurable n-gram size (min/max)
- **Character-level**: Used for substring matching
- **Token-level**: Used for phrase detection
- **Overlap**: Sliding window approach for generation