
.. image:: _static/lance_logo.png
  :width: 400

Lance: modern columnar format for ML workloads
==============================================


`Lance` is a columnar format that is easy and fast to version, query and train on.
It’s designed to be used with images, videos, 3D point clouds, audio and of course tabular data.
It supports any POSIX file systems, and cloud storage like AWS S3 and Google Cloud Storage.
The key features of Lance include:

* **High-performance random access:** 100x faster than Parquet.

* **Zero-copy schema evolution:** add and drop columns without copying the entire dataset.

* **Vector search:** find nearest neighbors in under 1 millisecond and combine OLAP-queries with vector search.

* **Ecosystem integrations:** Apache-Arrow, DuckDB and more on the way.


Installation
------------

You can install Lance via pip:

.. code-block:: bash

    pip install pylance

For the latest features and bug fixes, you can install the preview version:

.. code-block:: bash

    pip install --pre --extra-index-url https://pypi.fury.io/lancedb/ pylance

Preview releases receive the same level of testing as regular releases.


.. toctree::
   :caption: Introduction
   :maxdepth: 2

   Quickstart <./notebooks/quickstart>
   ./introduction/read_and_write
   ./introduction/schema_evolution

.. toctree::
   :caption: Advanced Usage
   :maxdepth: 1

   Lance Format Spec <./format>
   Blob API <./blob>
   Object Store Configuration <./object_store>
   Distributed Write <./distributed_write>
   Performance Guide <./performance>
   Tokenizer <./tokenizer>
   Extension Arrays <./arrays>

.. toctree::
   :caption: Integrations

   Huggingface <./integrations/huggingface>
   Tensorflow <./integrations/tensorflow>
   PyTorch <./integrations/pytorch>
   Ray <./integrations/ray>

.. toctree::
   :maxdepth: 1

   API References <./api/api>
   Contributor Guide <./contributing>
   Examples <./examples/examples>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
