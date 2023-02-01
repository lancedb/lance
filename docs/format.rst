File Format
===========

Dataset Directory
------------------

A `Lance Dataset` is organized in a directory.

.. code-block::

    /path/to/dataset:
        data/*.lance  -- Data directory
        latest.manifest -- The manifest file for the latest version.
        _versions/*.manifest -- Manifest file for each dataset version.
        _indices/{UUID-*}/index.idx -- Secondary index, each index per directory.


A ``Manifest`` file includes the metadata to describe a version of the dataset.

.. code-block:: protobuf

    // Manifest is a global section shared between all the files.
    message Manifest {
        // All fields of the dataset, including the nested fields.
        repeated Field fields = 1;

        // Fragments of the dataset.
        repeated DataFragment fragments = 2;

        // Snapshot version number.
        uint64 version = 3;

        // The file position of the version auxiliary data.
        //  * It is not inheritable between versions.
        //  * It is not loaded by default during query.
        uint64 version_aux_data = 4;

        // Schema metadata.
        map<string, bytes> metadata = 5;

        // If presented, the file position of the index metadata.
        optional uint64 index_section = 6;
    }

``DataFragment`` represents a chunk of data in the dataset. Itself includes one or more ``DataFile``,
where each ``DataFile`` can contain several columns in the chunk of data.

.. code-block:: protobuf

    // Data fragment. A fragment is a set of files which represent the
    // different columns of the same rows.
    // If column exists in the schema, but the related file does not exist,
    // treat this column as nulls.
    message DataFragment {
        // Unique ID of each DataFragment
        uint64 id = 1;

        repeated DataFile files = 2;
    }

    message DataFile {
        // Relative path to the root.
        string path = 1;
        // The ids of the fields/columns in this file
        repeated int32 fields = 2;
    }

File Structure
--------------

Each ``.lance`` file is the container for the actual data.

.. image:: file_struct.png

At the tail of the file, a `Metadata` protobuf block is used to describe the structure of the data file.

.. code-block:: protobuf

    message Metadata {
        uint64 manifest_position = 1;

        // Logical offsets of each chunk group, i.e., number of the rows in each
        // chunk.
        repeated int32 batch_offsets = 2;

        // The file position that page table is stored.
        //
        // A page table is a matrix of N x N x 2, where N = num_fields, and M =
        // num_batches. Each cell in the table is a pair of <position:uint64,
        // length:uint64> of the page. Both position and length are uint64 values. The
        // <position, length> of all the pages in the same column are then
        // contiguously stored.
        //
        // For example, for the column 5 and batch 4, we have:
        // ```text
        //   position = page_table[5][4][0];
        //   length = page_table[5][4][1];
        // ```
        uint64 page_table_position = 3;
    }

Optionally, a ``Manifest`` block can be stored after the ``Metadata`` block, to make the lance file self-describable.

In the end of the file, a ``Footer`` is written to indicate the closure of a file:

.. code-block::

    +---------------+----------------+
    | 0 - 3 byte    | 4 - 7 byte     |
    +===============+================+
    | metadata position (uint64)     |
    +---------------+----------------+
    | major version | minor version  |
    +---------------+----------------+
    |   Magic number "LANC"          |
    +--------------------------------+



Encodings
---------

`Lance` uses encodings that can render good both point query and scan performance.
Generally, it requires:

1. It takes no more than 2 disk reads to access any data points.
2. It takes sub-linear computation (``O(n)``) to locate one piece of data.

Plain Encoding
~~~~~~~~~~~~~~

Plain encoding stores Arrow array with **fixed size** values, such as primitive values, in contiguous space on disk.
Because the size of each value is fixed, the offset of a particular value can be computed directly.

Null: TBD

Variable-Length Binary Encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For variable-length data types, i.e., ``(Large)Binary / (Large)String / (Large)List`` in Arrow, Lance uses variable-length
encoding. Similar to Arrow in-memory layout, the on-disk layout include an offset array, and the actual data array.
The offset array contains the **absolute offset** of each value appears in the file.

.. code-block::

    +---------------+----------------+
    | offset array  | data array     |
    +---------------+----------------+


If ``offsets[i] == offsets[i + 1]``, we treat the ``i-th`` value as ``Null``.

Dictionary Encoding
~~~~~~~~~~~~~~~~~~~

Directory encoding is a composite encoding for a
`Arrow Dictionary Type <https://arrow.apache.org/docs/python/generated/pyarrow.DictionaryType.html#pyarrow.DictionaryType>`_,
where Lance encodes the `key` and `value` separately using primitive encoding types,
i.e., `key` are usually encoded with `Plain Encoding`_.


Dataset Update and Schema Evolution
-----------------------------------

``Lance`` supports fast dataset update and schema evolution via manipulating the ``Manifest`` metadata.

``Appending`` is done by appending new ``Fragment`` to the dataset.
While adding columns is done by adding new ``DataFile`` of the new columns to each ``Fragment``.
Finally, ``Overwrite`` a dataset can be done by resetting the ``Fragment`` list of the ``Manifest``.

.. image:: schema_evolution.png