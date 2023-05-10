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

    lance.write_dataset(producer, "./alice_and_bob.lance")

:py:meth:`lance.write_dataset` supports writing :py:class:`pyarrow.Table`, :py:class:`pandas.DataFrame`,
`pyarrow.Dataset`, and `Iterator[pyarrow.RecordBatch]`. Check the function doc for more details.

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

For example, the following filter string is acceptable:

  .. code-block:: SQL

    ((label IN [10, 20]) AND (note.email IS NOT NULL))
        OR NOT note.created

Random read
~~~~~~~~~~~

One district feature of Lance, as columnar format, is that it allows you to read random samples quickly.

    .. code-block:: python

        # Access the 2nd, 101th and 501th rows
        data = ds.take([1, 100, 500], columns=["image", "label"])

