Read and Write Lance Dataset
============================

Lance dataset APIs follows the `PyArrow API <https://arrow.apache.org/docs/python/parquet.html>`_
conventions.

Writing Lance Dataset
---------------------

Similar to Apache Pyarrow, the simplest approach to create a Lance dataset is
writing a :py:class:`pyarrow.Table` via :py:meth:`lance.write_dataset`.

  .. code-block:: python

    import lance
    import pyarrow as pa

    table = pa.Table.from_pylist([{"name": "Alice", "age": 20},
                                  {"name": "Bob", "age": 30}])
    lance.write_dataset(table, "./alice_and_bob.lance")

If the memory footprint of the dataset is too large to fit in memory, :py:meth:`lance.write_dataset`
also supports writing a dataset in iterator of :py:class:`pyarrow.RecordBatch` es.

  .. code-block:: python

    import lance
    import pyarrow as pa

    def producer():
        yield pa.RecordBatch.from_pylist([{"name": "Alice", "age": 20}])
        yield pa.RecordBatch.from_pylist([{"name": "Blob", "age": 30}])

    schema = pa.schema([
            pa.field("name", pa.string()),
            pa.field("age", pa.int64()),
        ])

    lance.write_dataset(reader, "./alice_and_bob.lance", schema)

:py:meth:`lance.write_dataset` supports writing :py:class:`pyarrow.Table`, :py:class:`pandas.DataFrame`,
:py:class:`pyarrow.Dataset`, and ``Iterator[pyarrow.RecordBatch]``. Check its doc for more details.

Adding new columns
~~~~~~~~~~~~~~~~~~

New columns can be merged into an existing dataset in using :py:meth:`lance.Dataset.merge`.
This allows filling in additional columns without having to rewrite the whole dataset.

To use the ``merge`` method, provide a new table that includes the columns you
want to add, and a column name to use for joining the new data to the existing
dataset.

For example, imagine we have a dataset of embeddings and ids:

.. testcode::

    import lance
    import pyarrow as pa
    import numpy as np
    table = pa.table({
       "id": pa.array([1, 2, 3]),
       "embedding": pa.array([np.array([1, 2, 3]), np.array([4, 5, 6]),
                              np.array([7, 8, 9])])
    })
    dataset = lance.write_dataset(table, "embeddings")

Now if we want to add a column of labels we have generated, we can do so by merging a new table:

.. testcode::

    new_data = pa.table({
       "id": pa.array([1, 2, 3]),
       "label": pa.array(["horse", "rabbit", "cat"])
    })
    dataset.merge(new_data, "id")
    dataset.to_table().to_pandas()

.. testoutput::

       id  embedding   label
    0   1  [1, 2, 3]   horse
    1   2  [4, 5, 6]  rabbit
    2   3  [7, 8, 9]     cat

Deleting rows
~~~~~~~~~~~~~

Lance supports deleting rows from a dataset using a SQL filter. For example, to
delete Bob's row from the dataset above, one could use:

  .. code-block:: python

    import lance

    dataset = lance.dataset("./alice_and_bob.lance")
    dataset.delete("name = 'Bob'")

:py:meth:`lance.LanceDataset.delete` supports the same filters as described in
:ref:`filter-push-down`.

Rows are deleted by marking them as deleted in a separate deletion index. This is
faster than rewriting the files and also avoids invaliding any indices that point
to those files. Any subsequent queries will not return the deleted rows.

.. warning::
  
  Do not read datasets with deleted rows using Lance versions prior to 0.5.0,
  as they will return the deleted rows. This is fixed in 0.5.0 and later.

Updating rows
~~~~~~~~~~~~~

Lance supports updating rows based on SQL expressions with the
:py:meth:`lance.LanceDataset.update` method. For example, if we notice
that Bob's name in our dataset has been sometimes written as ``Blob``, we can fix
that with:

  .. code-block:: python

    import lance

    dataset = lance.dataset("./alice_and_bob.lance")
    dataset.update({"name": "'Bob'"}), where="name = 'Blob'")

The update values are SQL expressions, which is why ``'Bob'`` is wrapped in single
quotes. This means we can use complex expressions that reference existing columns if
we wish. For example, if two years have passed and we wish to update the ages
of Alice and Bob in the same example, we could write:

  .. code-block:: python

    import lance

    dataset = lance.dataset("./alice_and_bob.lance")
    dataset.update({"age": "age + 2"})

.. TODO: Once we implement MERGE, we should make a note that this method shouldn't be used 
..       for updating single rows in a loop, and users should instead do bulk updates
..       using MERGE.

Committing mechanisms for S3
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most supported storage systems (e.g. local file system, Google Cloud Storage,
Azure Blob Store) natively support atomic commits, which prevent concurrent
writers from corrupting the dataset. However, S3 does not support this natively.
To work around this, you may provide a locking mechanism that Lance can use to
lock the table while providing a write. To do so, you should implement a
context manager that acquires and releases a lock and then pass that to the
``commit_lock`` parameter of :py:meth:`lance.write_dataset`.

.. note::

  In order for the locking mechanism to work, all writers must use the same exact
  mechanism. Otherwise, Lance will not be able to detect conflicts.

On entering, the context manager should acquire the lock on the table. The table
version being committed is passed in as an argument, which may be used if the
locking service wishes to keep track of the current version of the table, but
this is not required. If the table is already locked by another transaction,
it should wait until it is unlocked, since the other transaction may fail. Once
unlocked, it should either lock the table or, if the lock keeps track of the
current version of the table, return a :class:`CommitConflictError` if the
requested version has already been committed.

To prevent poisoned locks, it's recommended to set a timeout on the locks. That
way, if a process crashes while holding the lock, the lock will be released
eventually. The timeout should be no less than 30 seconds.

.. code-block:: python

  from contextlib import contextmanager

  @contextmanager
  def commit_lock(version: int);
      # Acquire the lock
      my_lock.acquire()
      try:
        yield
      except:
        failed = True
      finally:
        my_lock.release()
  
  lance.write_dataset(data, "s3://bucket/path/", commit_lock=commit_lock)

When the context manager is exited, it will raise an exception if the commit
failed. This might be because of a network error or if the version has already
been written. Either way, the context manager should release the lock. Use a 
try/finally block to ensure that the lock is released.

Concurrent Writer on S3 using DynamoDB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

  This feature is experimental at the moment

Lance has native support for concurrent writers on S3 using DynamoDB instead of locking.
User may pass in a DynamoDB table name alone with the S3 URI to their dataset to enable this feature.

.. code-block:: python

  import lance
  # s3+ddb:// URL scheme let's lance know that you want to use DynamoDB for writing to S3 concurrently
  ds = lance.dataset("s3+ddb://my-bucket/mydataset.lance?ddbTableName=mytable")

The DynamoDB table is expected to have a primary hash key of ``base_uri`` and a range key ``version``.
The key ``base_uri`` should be string type, and the key ``version`` should be number type.

For details on how this feature works, please see :ref:`external-manifest-store`.


Reading Lance Dataset
---------------------

To open a Lance dataset, use the :py:meth:`lance.dataset` function:

  .. code-block:: python

    import lance
    ds = lance.dataset("s3://bucket/path/imagenet.lance")
    # Or local path
    ds = lance.dataset("./imagenet.lance")

  .. note::

    Lance supports local file system, AWS ``s3`` and Google Cloud Storage(``gs``) as storage backends
    at the moment. See :ref:`storages` for more details.

The most straightforward approach for reading a Lance dataset is to utilize the :py:meth:`lance.LanceDataset.to_table`
method in order to load the entire dataset into memory.

  .. code-block:: python

    table = ds.to_table()

Due to Lance being a high-performance columnar format, it enables efficient reading of subsets of the dataset by utilizing
**Column (projection)** push-down and **filter (predicates)** push-downs.

    .. code-block:: python

        table = ds.to_table(
            columns=["image", "label"],
            filter="label = 2 AND text IS NOT NULL",
            limit=1000,
            offset=3000)

Lance understands the cost of reading heavy columns such as ``image``.
Consequently, it employs an optimized query plan to execute the operation efficiently.

Iterative Read
~~~~~~~~~~~~~~

If the dataset is too large to fit in memory, you can read it in batches
using the :py:meth:`lance.LanceDataset.to_batches` method:

  .. code-block:: python

    for batch in ds.to_batches(columns=["image"], filter="label = 10"):
        # do something with batch
        compute_on_batch(batch)

Unsurprisingly, :py:meth:`~lance.LanceDataset.to_batches` takes the same parameters
as :py:meth:`~lance.LanceDataset.to_table` function.


.. _filter-push-down:

Filter push-down
~~~~~~~~~~~~~~~~

Lance embraces the utilization of standard SQL expressions as predicates for dataset filtering.
By pushing down the SQL predicates directly to the storage system,
the overall I/O load during a scan is significantly reduced.

Currently, Lance supports a growing list of expressions.

* ``>``, ``>=``, ``<``, ``<=``, ``=``
* ``AND``, ``OR``, ``NOT``
* ``IS NULL``, ``IS NOT NULL``
* ``IS TRUE``, ``IS NOT TRUE``, ``IS FALSE``, ``IS NOT FALSE``
* ``IN``
* ``LIKE``, ``NOT LIKE``
* ``regexp_match(column, pattern)``
* ``CAST``

For example, the following filter string is acceptable:

  .. code-block:: SQL

    ((label IN [10, 20]) AND (note.email IS NOT NULL))
        OR NOT note.created

If your column name contains special characters or is a `SQL Keyword <https://docs.rs/sqlparser/latest/sqlparser/keywords/index.html>`_,
you can use backtick (`````) to escape it. For nested fields, each segment of the
path must be wrapped in backticks. 

  .. code-block:: SQL

    `CUBE` = 10 AND `column name with space` IS NOT NULL
      AND `nested with space`.`inner with space` < 2

.. warning::

  Field names containing periods (``.``) are not supported.

Literals for dates, timestamps, and decimals can be written by writing the string
value after the type name. For example

  .. code-block:: SQL

    date_col = date '2021-01-01'
    and timestamp_col = timestamp '2021-01-01 00:00:00'
    and decimal_col = decimal(8,3) '1.000'

For timestamp columns, the precision can be specified as a number in the type
parameter. Microsecond precision (6) is the default.

.. list-table::
    :widths: 30 40
    :header-rows: 1

    * - SQL
      - Time unit
    * - ``timestamp(0)``
      - Seconds
    * - ``timestamp(3)``
      - Milliseconds
    * - ``timestamp(6)``
      - Microseconds
    * - ``timestamp(9)``
      - Nanoseconds

Lance internally stores data in Arrow format. The mapping from SQL types to Arrow
is:

.. list-table::
    :widths: 30 40
    :header-rows: 1

    * - SQL type
      - Arrow type
    * - ``boolean``
      - ``Boolean``
    * - ``tinyint`` / ``tinyint unsigned``
      - ``Int8`` / ``UInt8``
    * - ``smallint`` / ``smallint unsigned``
      - ``Int16`` / ``UInt16``
    * - ``int`` or ``integer`` / ``int unsigned`` or ``integer unsigned``
      - ``Int32`` / ``UInt32``
    * - ``bigint`` / ``bigint unsigned``
      - ``Int64`` / ``UInt64``
    * - ``float``
      - ``Float32``
    * - ``double``
      - ``Float64``
    * - ``decimal(precision, scale)``
      - ``Decimal128``
    * - ``date``
      - ``Date32``
    * - ``timestamp``
      - ``Timestamp`` (1)
    * - ``string``
      - ``Utf8``
    * - ``binary``
      - ``Binary``

(1) See precision mapping in previous table.


Random read
~~~~~~~~~~~

One district feature of Lance, as columnar format, is that it allows you to read random samples quickly.

    .. code-block:: python

        # Access the 2nd, 101th and 501th rows
        data = ds.take([1, 100, 500], columns=["image", "label"])

The ability to achieve fast random access to individual rows plays a crucial role in facilitating various workflows
such as random sampling and shuffling in ML training.
Additionally, it empowers users to construct secondary indices,
enabling swift execution of queries for enhanced performance.


Table Maintenance
-----------------

Some operations over time will cause a Lance dataset to have a poor layout. For
example, many small appends will lead to a large number of small fragments. Or
deleting many rows will lead to slower queries due to the need to filter out
deleted rows.

To address this, Lance provides methods for optimizing dataset layout.

Compact data files
~~~~~~~~~~~~~~~~~~

Data files can be rewritten so there are fewer files. When passing a
``target_rows_per_fragment`` to :py:meth:`lance.dataset.DatasetOptimizer.compact_files`,
Lance will skip any fragments that are already above that row count, and rewrite
others. Fragments will be merged according to their fragment ids, so the inherent
ordering of the data will be preserved.

.. note::

  Compaction creates a new version of the table. It does not delete the old 
  version of the table and the files referenced by it.

.. code-block:: python

    import lance

    dataset = lance.dataset("./alice_and_bob.lance")
    dataset.optimize.compact_files(target_rows_per_fragment=1024 * 1024)

During compaction, Lance can also remove deleted rows. Rewritten fragments will
not have deletion files. This can improve scan performance since the soft deleted
rows don't have to be skipped during the scan.

When files are rewritten, the original row ids are invalidated. This means the
affected files are no longer part of any ANN index if they were before. Because
of this, it's recommended to rewrite files before re-building indices.
