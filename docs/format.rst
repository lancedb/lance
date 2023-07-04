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

.. literalinclude:: ../protos/format.proto
   :language: protobuf
   :linenos:
   :start-at: // Manifest is
   :end-at: } // Manifest

``DataFragment`` represents a chunk of data in the dataset. Itself includes one or more ``DataFile``,
where each ``DataFile`` can contain several columns in the chunk of data. It also may include a 
``DeletionFile``, which is explained in a later section.

.. literalinclude:: ../protos/format.proto
   :language: protobuf
   :linenos:
   :start-at: // Data fragment
   :end-at: } // DataFile


File Structure
--------------

Each ``.lance`` file is the container for the actual data.

.. image:: file_struct.png

At the tail of the file, a `Metadata` protobuf block is used to describe the structure of the data file.

.. literalinclude:: ../protos/format.proto
   :language: protobuf
   :linenos:
   :start-at: message Metadata {
   :end-at: } // Metadata

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

Feature Flags
-------------

As the file format and dataset evolve, new feature flags are added to the
format. There are two separate fields for checking for feature flags, depending
on whether you are trying to read or write the table. Readers should check the
``reader_feature_flags`` to see if there are any flag it is not aware of. Writers 
should check ``writer_feature_flags``. If either sees a flag they don't know, they
should return an "unsupported" error on any read or write operation.

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


Deletion
--------

Rows can be marked deleted by adding a deletion file next to the data in the
``_deletions`` folder. These files contain the indices of rows that have between
deleted for some fragment. For a given version of the dataset, each fragment can
have up to one deletion file. Fragments that have no deleted rows have no deletion
file.

Readers should filter out row ids contained in these deletion files during a 
scan or ANN search.

Deletion files come in two flavors:

1. Arrow files: which store a column with a flat vector of indices
2. Roaring bitmaps: which store the indices as compressed bitmaps.

`Roaring Bitmaps`_ are used for larger deletion sets, while Arrow files are used for
small ones. This is because Roaring Bitmaps are known to be inefficient for small
sets.


.. _`Roaring Bitmaps`: https://roaringbitmap.org/

The filenames of deletion files are structured like:

.. code-block::

    _deletions/{fragment_id}-{read_version}-{random_id}.{arrow|bin}

Where ``fragment_id`` is the fragment the file corresponds to, ``read_version`` is
the version of the dataset that it was created off of (usually one less than the
version it was committed to), and ``random_id`` is a random i64 used to avoid
collisions. The suffix is determined by the file type (``.arrow`` for Arrow file,
``.bin`` for roaring bitmap).

.. literalinclude:: ../protos/format.proto
   :language: protobuf
   :linenos:
   :start-at: // Deletion File
   :end-at: } // DeletionFile

Deletes can be materialized by re-writing data files with the deleted rows 
removed. However, this invalidates row indices and thus the ANN indices, which
can be expensive to recompute.
