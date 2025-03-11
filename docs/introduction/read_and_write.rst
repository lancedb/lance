Read and Write Data
===================

Writing Lance Dataset
---------------------

If you're familiar with `Apache PyArrow <https://arrow.apache.org/docs/python/getstarted.html>`_,
you'll find that creating a Lance dataset is straightforward.
Begin by writing a :py:class:`pyarrow.Table` using the :py:meth:`lance.write_dataset` function.

.. testsetup::

  shutil.rmtree("./alice_and_bob.lance", ignore_errors=True)

.. doctest::

  >>> import lance
  >>> import pyarrow as pa

  >>> table = pa.Table.from_pylist([{"name": "Alice", "age": 20},
  ...                               {"name": "Bob", "age": 30}])
  >>> ds = lance.write_dataset(table, "./alice_and_bob.lance")

If the dataset is too large to fully load into memory, you can stream data using :py:meth:`lance.write_dataset`
also supports :py:class:`~typing.Iterator` of :py:class:`pyarrow.RecordBatch` es.
You will need to provide a :py:class:`pyarrow.Schema` for the dataset in this case.

.. testsetup:: rst_generator

  shutil.rmtree("./alice_and_bob.lance", ignore_errors=True)

.. doctest:: rst_generator

  >>> def producer() -> Iterator[pa.RecordBatch]:
  ...     """An iterator of RecordBatches."""
  ...     yield pa.RecordBatch.from_pylist([{"name": "Alice", "age": 20}])
  ...     yield pa.RecordBatch.from_pylist([{"name": "Bob", "age": 30}])

  >>> schema = pa.schema([
  ...     ("name", pa.string()),
  ...     ("age", pa.int32()),
  ... ])

  >>> ds = lance.write_dataset(producer(),
  ...                          "./alice_and_bob.lance",
  ...                          schema=schema, mode="overwrite")
  >>> ds.count_rows()
  2

:py:meth:`lance.write_dataset` supports writing :py:class:`pyarrow.Table`, :py:class:`pandas.DataFrame`,
:py:class:`pyarrow.dataset.Dataset`, and ``Iterator[pyarrow.RecordBatch]``.

Adding Rows
-----------

To insert data into your dataset, you can use either :py:meth:`LanceDataset.insert <lance.LanceDataset.insert>`
or :py:meth:`~lance.write_dataset` with ``mode=append``.

.. testsetup::

  shutil.rmtree("./insert_example.lance", ignore_errors=True)

.. doctest::

  >>> import lance
  >>> import pyarrow as pa

  >>> table = pa.Table.from_pylist([{"name": "Alice", "age": 20},
  ...                               {"name": "Bob", "age": 30}])
  >>> ds = lance.write_dataset(table, "./insert_example.lance")

  >>> new_table = pa.Table.from_pylist([{"name": "Carla", "age": 37}])
  >>> ds.insert(new_table)
  >>> tbl = lance.dataset("./insert_example.lance").to_table()
  >>> tbl.to_pandas()
      name  age
  0  Alice   20
  1    Bob   30
  2  Carla   37

  >>> new_table2 = pa.Table.from_pylist([{"name": "David", "age": 42}])
  >>> ds2 = lance.write_dataset(new_table2, "./insert_example.lance", mode="append")
  >>> ds2.to_table().to_pandas()
      name  age
  0  Alice   20
  1    Bob   30
  2  Carla   37
  3  David   42


Deleting rows
-------------

Lance supports deleting rows from a dataset using a SQL filter, as described in :ref:`filter-push-down`.
For example, to delete Bob's row from the dataset above, one could use:

.. doctest::

  >>> import lance

  >>> dataset = lance.dataset("./alice_and_bob.lance")
  >>> dataset.delete("name = 'Bob'")
  >>> dataset2 = lance.dataset("./alice_and_bob.lance")
  >>> dataset2.to_table().to_pandas()
      name  age
  0  Alice   20


.. note::

  :doc:`Lance Format is immutable <../format>`. Each write operation creates a new version of the dataset,
  so users must reopen the dataset to see the changes. Likewise, rows are removed by marking
  them as deleted in a separate deletion index, rather than rewriting the files. This approach
  is faster and avoids invalidating any indices that reference the files, ensuring that subsequent
  queries do not return the deleted rows.


Updating rows
-------------

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

If you are trying to update a set of individual rows with new values then it is often
more efficient to use the merge insert operation described below.

.. code-block:: python

  import lance

  # Change the ages of both Alice and Bob
  new_table = pa.Table.from_pylist([{"name": "Alice", "age": 30},
                                    {"name": "Bob", "age": 20}])

  # This works, but is inefficient, see below for a better approach
  dataset = lance.dataset("./alice_and_bob.lance")
  for idx in range(new_table.num_rows):
    name = new_table[0][idx].as_py()
    new_age = new_table[1][idx].as_py()
    dataset.update({"age": new_age}, where=f"name='{name}'")

Merge Insert
------------

Lance supports a merge insert operation.  This can be used to add new data in bulk
while also (potentially) matching against existing data.  This operation can be used
for a number of different use cases.

Bulk Update
^^^^^^^^^^^

The :py:meth:`lance.LanceDataset.update` method is useful for updating rows based on
a filter.  However, if we want to replace existing rows with new rows then a merge
insert operation would be more efficient:

.. code-block:: python

  import lance

  # Change the ages of both Alice and Bob
  new_table = pa.Table.from_pylist([{"name": "Alice", "age": 30},
                                    {"name": "Bob", "age": 20}])
  dataset = lance.dataset("./alice_and_bob.lance")
  # This will use `name` as the key for matching rows.  Merge insert
  # uses a JOIN internally and so you typically want this column to
  # be a unique key or id of some kind.
  dataset.merge_insert("name") \
         .when_matched_update_all() \
         .execute(new_table)

Note that, similar to the update operation, rows that are modified will
be removed and inserted back into the table, changing their position to
the end.  Also, the relative order of these rows could change because we
are using a hash-join operation internally.

Insert if not Exists
^^^^^^^^^^^^^^^^^^^^

Sometimes we only want to insert data if we haven't already inserted it
before.  This can happen, for example, when we have a batch of data but
we don't know which rows we've added previously and we don't want to
create duplicate rows.  We can use the merge insert operation to achieve
this:

.. code-block:: python

  import lance

  # Bob is already in the table, but Carla is new
  new_table = pa.Table.from_pylist([{"name": "Bob", "age": 30},
                                    {"name": "Carla", "age": 37}])

  dataset = lance.dataset("./alice_and_bob.lance")

  # This will insert Carla but leave Bob unchanged
  dataset.merge_insert("name") \
         .when_not_matched_insert_all() \
         .execute(new_table)

Update or Insert (Upsert)
^^^^^^^^^^^^^^^^^^^^^^^^^

Sometimes we want to combine both of the above behaviors.  If a row
already exists we want to update it.  If the row does not exist we want
to add it.  This operation is sometimes called "upsert".  We can use
the merge insert operation to do this as well:

.. code-block:: python

  import lance

  # Change Carla's age and insert David
  new_table = pa.Table.from_pylist([{"name": "Carla", "age": 27},
                                    {"name": "David", "age": 42}])

  dataset = lance.dataset("./alice_and_bob.lance")

  # This will update Carla and insert David
  dataset.merge_insert("name") \
         .when_matched_update_all() \
         .when_not_matched_insert_all() \
         .execute(new_table)

Replace a Portion of Data
^^^^^^^^^^^^^^^^^^^^^^^^^

A less common, but still useful, behavior can be to replace some region
of existing rows (defined by a filter) with new data.  This is similar
to performing both a delete and an insert in a single transaction.  For
example:

.. code-block:: python

  import lance

  new_table = pa.Table.from_pylist([{"name": "Edgar", "age": 46},
                                    {"name": "Francene", "age": 44}])

  dataset = lance.dataset("./alice_and_bob.lance")

  # This will remove anyone above 40 and insert our new data
  dataset.merge_insert("name") \
         .when_not_matched_insert_all() \
         .when_not_matched_by_source_delete("age >= 40") \
         .execute(new_table)



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
  at the moment. Read more in `Object Store Configuration`_.

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

  ((label IN [10, 20]) AND (note['email'] IS NOT NULL))
      OR NOT note['created']

Nested fields can be accessed using the subscripts. Struct fields can be
subscripted using field names, while list fields can be subscripted using
indices.

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

When files are rewritten, the original row addresses are invalidated. This means the
affected files are no longer part of any ANN index if they were before. Because
of this, it's recommended to rewrite files before re-building indices.

.. TODO: remove this last comment once move-stable row ids are default.
