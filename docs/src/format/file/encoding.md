# Lance Encoding Strategy

The encoding strategy determines how array data is encoded into a disk page. The encoding strategy tends to evolve
more quickly than the file format itself.

## Older Encoding Strategies

The 0.1 and 2.0 encoding strategies are no longer documented. They were significantly different from future encoding
strategies and describing them in detail would be a distraction.

## Terminology

An array is a sequence of values. An array has a data type which describes the semantic interpretation of the values.
A layout is a way to encode an array into a set of buffers and child arrays. A buffer is a contiguous sequence of
bytes. An encoding describes how the semantic interpretation of data is mapped to the layout. An encoder converts
data from one encoding to another.

Data types and layouts are orthogonal concepts. An integer array might be stored as a fixed width layout with a plain
encoding or as a variable width layout with a variable-width integer encoding. Layouts can contain other layouts
which allows for a tree of encodings to be built. For example, an string array might be encoded with dictionary
encoding where the indices are encoded with run length encoding where the run lengths are encoded with bitpacking
and the packed bits are encoded with plain encoding. Meanwhile the run values are encoded with a variable width
integer encoding. Meanwhile the dictionary values are encoded with FSST encoding and the FSST strings are encoded
with binary encoding.

### Data Types

Lance uses a subset of Arrow's type system for data types. An Arrow data type is is both a data type and an encoding.
When writing data Lance will often normalize Arrow data types. For example, a string array and a large string array
might end up traveling down the same path (variable width data). In fact, most types fall into two general paths. One
for fixed-width data and one for variable-width data (where we recognize both u32 and u64 offsets).

At read time, the Arrow data type is used to determine the target encoding. For example, a string array and large
string array might both be stored in the same layout but, at read time, we will use the Arrow data type to determine
the size of the offsets returned to the user. There is no requirement the output Arrow type matches the input Arrow
type. For example, it is acceptable to write an array as "large string" and then read it back as "string".

### Layouts

Layouts describe how data is encoded into buffers and child arrays. Encoders generally expect data to be in some
kind of input layout and will generate some kind of output layout. For example, an FSST encoder expects the input
to be in a variable width layout and will generate a variable width layout as output.


| Layout Name    | Description                                                                                                                                                                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fixed Width    | A fixed width layout is a layout where each value has a fixed number of bits and the array is encoded into a single buffer.                                                                                                                                       |
| Variable Width | A variable width layout is a layout where each value has a variable number of bytes. The array is encoded into one buffer of values and one array of offsets. The offsets are semantically either u32 or u64 (the actual encoding of the offsets may be smaller). |
| Dictionary     | A dictionary layout is a layout where the data is split into a dictionary array and an indices array. The indices array contains indices into the dictionary array. When Lance encodes dictionaries it uses u32 indices.                                          |
| Opaque         | An opaque layout is a layout where the data is encoded into a single buffer and it is impossible to know where any single value is located. For example, this is the output of something like GZIP compression.                                                   |
| Struct         | A struct array is a layout where each struct child is encoded into a separate array.                                                                                                                                                                              |
| Packed Struct  | A packed struct layout is a layout where all struct children are encoded into a single array.                                                                                                                                                                     |

## Structural Encoding

Structural encoding defines how an arrays compressed buffers are eventually stored on disk.  The choice of structural
encoding determines what sort of compression we can use.  Structural encoding is also the point where we handle validity
and repetition (i.e. lists).  Structural encoding defines what level of column projection and row selection is available.
Finally, structural encoding defines how we map row ranges into byte ranges, regardless of the compression that is used.

### Struct & List Encodings

Struct and list information is encoded into repetition and definition levels.  This gives us a single buffer of repdef
levels which has roughly the same length as the flattened array.  Everything that is not a struct or list is a primitive
data type.  Fixed size lists of primitive fields are themselves primitive.  Fixed size lists of non-primitive fields are
not primitive.

### Mini Block Encoding

Mini block encoding encodes small primitive types (the current threshold is 128 bytes) into small blocks of data.  Each
block is opaque.  The entire block must be decoded in order to access any single value.  Mini blocks are generally 1 to
2 disk sectors (4KiB - 8KiB) and so the read amplification for reading a single mini block is trivial.

Metadata keeps track of how many items are in each block and the size of the block in bytes.  This metadata is small (2
bytes per block) and intended to be cached in memory to accelerate point lookups.  Once this metadata is cached then
any value can be loaded with a single IOP.  Since mini blocks are opaque we do not bother zipping together the different
compressed buffers.  This means we can use any type of compression including non-transparent compression such as delta
compression.

### Full Zip Encoding

Full zip encoding encodes large primitive types (e.g. vector embeddings, audio, images, prompts, etc.)  These types are
not well suited for mini block encoding since each value would occupy an entire block overhead and RAM requirements are
too large for many cases.  If the value is variable-width then an additional repetition index is written which is a set
of offsets into the position of each value.  This repetition index can be cached but is not by default.  This means that
we require 2 IOPS to access any value, one to fetch the offsets from the repetition index and a second to fetch the value.

Full zip encoding compresses an entire disk page at a time.  This allows compression to take advantage of patterns in the
data (such as similar images) which would not be possible in a mini block approach.  However, since the data must be zipped
together after compression we do require that transparent compression techniques be used.  This limits the available
compression techniques.

## Compression

After we choose a structural encoding we much choose how we will compress the data.  This is done by examining the data to
determine what patterns are present and then choosing an appropriate compression strategy.  By default we do not use general
purpose compression such as gzip or lz4 in most cases because these heavyweight compression algorithms often incur too high
of a CPU cost at read time.  Instead we aim for lightweight compression techniques.  The set of compression techniques available
and the rules used to decide on compression change from version to version.

### 2.1 Compression Strategy

### Summary of Compression Techniques

The following is a short summary of the compression techniques currently utilized.  The details of how a compression technique is used
will differ based on the data type and structural encoding chosen.

| Encoding | When it is used | Description |
| --- | --- | --- |
| Flat | Fixed width data when no other technique applies | Encodes a fixed-width layout into a single contiguous buffer |
| Binary | Variable width data when no other technique applies | Encodes a variable-width layout into a single contiguous buffer of data and a child fixed-width layout of offsets |
| Bitpacking | Fixed width data where not all bits are used | Compresses a fixed-width layout into a fixed-width layout with fewer bits per value by throwing away unused bits |
| Dictionary | Data with few unique values | Compresses fixed-width or variable-width layouts by splitting data into a dictionary (with the same layout as the input) with one row per unique value and a fixed-width layout of indices with the same number of rows as the input. |
| Fsst | Variable width data with repeated sequences | Compresses a variable-width layout into a smaller variable-width layout and a symbol table.  The symbol table is a small dictionary referred to by the data |
| Run Length Encoding | Data with runs of equal values | Compresses fixed-width or variable-width layouts by splitting data into an array of values (with the same layout as the input) and a fixed-width layout of run lengths. |
| Byte Stream Split | Fixed width data with byte-specific entropy patterns | Compresses (typically floating point) fixed-width data by splitting a stream of N-byte words into N streams of bytes. |
| General | When compression should be maximized or patterns are not obvious | Applies a general compression strategy (e.g. gzip, lz4, zstd) to a fixed-width or variable-width layout and creates a single opaque buffer. |
