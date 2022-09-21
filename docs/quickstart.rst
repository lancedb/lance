Quick Start
===========

We've provided Linux and MacOS wheels for Lance in PyPI. You can install Lance python bindings via:

.. code-block:: bash

    pip install pylance

Exploratory Data Analysis
-------------------------

Thanks for its Apache Arrow-first APIs, `lance`` can be used as a native Arrow extension.
For example, it enables users to directly use DuckDB to analyze lance dataset via DuckDB's Arrow integration.

.. code-block:: python

    # pip install pylance duckdb
    import lance
    import duckdb


Understand Label distribution of Oxford Pet Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    ds = lance.dataset("s3://eto-public/datasets/oxford_pet/pet.lance")
    duckdb.query("select label, count(1) from ds group by label").to_arrow_table()


Model Training and Evaluation
-----------------------------

