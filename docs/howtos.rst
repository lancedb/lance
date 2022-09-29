How-To Guides
=============

.. toctree::
   :maxdepth: 1

    Dataset Analysis <./notebooks/label>

Label Analyze
-------------

Let's take `The Oxford-IIIT Pet Dataset <https://www.robots.ox.ac.uk/~vgg/data/pets/>`_
as example, to explore and understand the dataset using `Lance` and `DuckDB <https://duckdb.org/>`_.

.. testcode::

    import lance
    import duckdb
    import pyarrow as pa

    dataset = lance.dataset(
        "s3://eto-public/datasets/oxford_pet/oxford_pet.lance",
    )

Take a look of the dataset schema to understand what data we have.

.. doctest::

    >>> dataset.schema
    filename: string
    class: dictionary<values=string, indices=int8, ordered=0>
    species: dictionary<values=string, indices=int8, ordered=0>
    breed: int16
    split: dictionary<values=string, indices=int8, ordered=0>
    folder: string
    source: struct<database: string, annotation: string, image: string>
      child 0, database: string
      child 1, annotation: string
      child 2, image: string
    size: struct<width: int32, height: int32, depth: uint8>
      child 0, width: int32
      child 1, height: int32
      child 2, depth: uint8
    segmented: bool
    object: list<item: struct<name: dictionary<values=string, indices=uint8, ordered=0>, pose: dictionary<values=string, indices=uint8, ordered=0>, truncated: bool, occluded: bool, bndbox: struct<xmin: int32, ymin: int32, xmax: int32, ymax: int32>, difficult: bool>>
      child 0, item: struct<name: dictionary<values=string, indices=uint8, ordered=0>, pose: dictionary<values=string, indices=uint8, ordered=0>, truncated: bool, occluded: bool, bndbox: struct<xmin: int32, ymin: int32, xmax: int32, ymax: int32>, difficult: bool>
          child 0, name: dictionary<values=string, indices=uint8, ordered=0>
          child 1, pose: dictionary<values=string, indices=uint8, ordered=0>
          child 2, truncated: bool
          child 3, occluded: bool
          child 4, bndbox: struct<xmin: int32, ymin: int32, xmax: int32, ymax: int32>
              child 0, xmin: int32
              child 1, ymin: int32
              child 2, xmax: int32
              child 3, ymax: int32
          child 5, difficult: bool
    image: binary

Calculate label distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

    duckdb.query(
        "SELECT count(1) as cnt, class FROM dataset GROUP BY 2 ORDER BY class")


Calculate Label Distribution among splits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. testcode::

    print(duckdb.query("""
        SELECT
            count(1) as cnt, class, split
        FROM dataset
        GROUP BY 3, 2 ORDER BY class, split
    """).to_df())


.. testoutput::
    :options: +NORMALIZE_WHITESPACE

         cnt              class  split
    0     98         Abyssinian   test
    1     50         Abyssinian  train
    2     50         Abyssinian    val
    3    100             Bengal   test
    4     50             Bengal  train
    ..   ...                ...    ...
    106   50    wheaten_terrier  train
    107   50    wheaten_terrier    val
    108  100  yorkshire_terrier   test
    109   50  yorkshire_terrier  train
    110   50  yorkshire_terrier    val

    [111 rows x 3 columns]

