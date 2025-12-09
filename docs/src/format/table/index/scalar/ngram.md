# N-gram Index

N-gram indices break text into overlapping sequences (trigrams) for efficient substring matching.
They provide fast text search by indexing all 3-character sequences in the text after
applying ASCII folding and lowercasing.

## Index Details

```protobuf
%%% proto.message.NGramIndexDetails %%%
```

## Storage Layout

The N-gram index stores tokenized text as trigrams with their posting lists:

1. `ngram_postings.lance` - Trigram tokens and their posting lists

### File Schema

| Column         | Type   | Nullable | Description                                       |
|----------------|--------|----------|---------------------------------------------------|
| `tokens`       | UInt32 | true     | Hashed trigram token                              |
| `posting_list` | Binary | false    | Compressed bitmap of row IDs containing the token |

## Accelerated Queries

The N-gram index provides inexact results for the following query types:

| Query Type     | Description              | Operation                                             | Result Type |
|----------------|--------------------------|-------------------------------------------------------|-------------|
| **contains**   | Substring search in text | Finds all trigrams in query, intersects posting lists | AtMost      |