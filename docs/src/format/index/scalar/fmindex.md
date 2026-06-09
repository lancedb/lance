# FM-Index (Full-text / Substring / Regex Search)

The FM-Index (Ferragina-Manzini Index) is a compressed substring index based on the Burrows-Wheeler Transform (BWT). Unlike traditional inverted indexes (Full-Text Search) which index distinct words, the FM-Index enables efficient **arbitrary substring search**, **prefix match**, and **suffix/regular-expression search** directly on raw bytes.

In Lance, the FM-Index is designed to scale dynamically across millions of documents or large-scale datasets, and is partitioned using Lance's **Segmented Index** architecture to support incremental appends, disjoint fragment tracking, and segment merging.

## High-Level Architecture

The FM-Index indexes raw text by treating columns of strings or binary payloads as raw byte arrays. 

```
                     +----------------------------------------+
                     |            Lance Dataset               |
                     |   (Disjoint groups of Fragments 0..N)   |
                     +----------------------------------------+
                                         |
                       Divide fragments into num_segments
                                         |
                                         v
                     +----------------------------------------+
                     |            Segmented Index             |
                     |  +-----------+ +-----------+ +-------+ |
                     |  | Segment 1 | | Segment 2 | | ...   | |
                     |  |  (FM-Idx) | |  (FM-Idx) | |       | |
                     |  +-----------+ +-----------+ +-------+ |
                     +----------------------------------------+
```

Each segment contains its own self-contained physical FM-Index mapping byte sub-sequences to Lance global row IDs.

## Data Normalization & Sanitization

The FM-Index is **normalization-independent by design** because it operates entirely on raw bytes. 

### Byte Sanitization vs. Text Normalization

1. **Byte Sanitization (Core Index Layer)**:
   The physical FM-Index uses specific sentinel bytes internally to mark boundaries:
   - `\x00` is reserved as the global Burrows-Wheeler Transform (BWT) terminator character.
   - `\xFF` is reserved as the document/row separator character.
   
   To avoid breaking the indexing structures, any incoming occurrences of `\x00` or `\xFF` are sanitized by remapping them to space (`\x20`) characters at index-build time. No other bytes are changed in this layer.

2. **Text Normalization (User/Application Layer)**:
   Because the index faithfully maps raw bytes, any semantic normalization (such as case folding `Hello` -> `hello`, Unicode NFKC normalization, stemming, or whitespace collapsing) is fully decoupled from the core index engine:
   - To build a case-insensitive search index, users apply a lowercase transform to the column *prior* to indexing.
   - When querying, the user's query text must undergo the exact same normalization pipeline.

## Configurable Segment Partitioning

Merging or appending to BWT-based indexes cannot be done via simple concatenation; the BWT suffix array must be reconstructed by re-reading the text and rebuilding. To balance build cost and search performance, Lance allows configuring how fragments map to index segments.

- **`num_segments` parameter**: Configured at index-creation time. If `num_segments` is specified (e.g. `num_segments = 4`), Lance splits the target dataset fragments into disjoint subsets and builds independent FM-Index segments over each chunk.
- **Unindexed Appends**: When new fragments are appended to the dataset, a subsequent `create_index` execution with unindexed fragment coverage will construct a new separate segment representing only those new fragments, keeping existing segments fully intact.
- **Segment Merging**: Multiple existing index segments can be merged into a single segment under Lance's `merge_segments` protocol. Lance unions the fragment coverage bitmaps of the selected segments, re-reads the raw text from those covered fragments, and constructs a fresh unified FM-Index.

## Query Evaluation

When a substring query is submitted (e.g., `CONTAINS(column, "query_string")`):
1. The search string is sanitized (remapping any `\x00` or `\xFF` to spaces) and optionally normalized if the target index is normalized.
2. The query is dispatched across all active segments in the logical index in parallel.
3. Each segment performs a BWT backward-search to locate occurrences of the pattern.
4. Matching offsets are mapped back to absolute dataset Row IDs.
5. Results from all segments are unioned to produce the final selection.
