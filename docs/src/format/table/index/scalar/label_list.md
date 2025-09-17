# Label List Index

Label list indices are optimized for columns containing multiple labels or tags per row.

## Index Details

```protobuf
%%% proto.message.LabelListIndexDetails %%%
```

## Storage Layout

The label list index uses the inverted index implementation with specialized handling for multi-value columns.

### File Structure

Uses the same structure as the inverted index:
- Token files for unique labels
- Posting lists for label-to-row mappings
- Document metadata for row information

## Implementation Details

- **Multi-value Support**: Each row can have multiple labels
- **Set Operations**: Efficient AND/OR operations on label sets
- **Storage**: Uses inverted index infrastructure
- **Query Types**: Supports contains-any, contains-all, and exact match queries