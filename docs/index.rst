
.. image:: _static/lance_logo.png
  :width: 400

Lance: modern columnar data format for ML
======================================================================================


`Lance` is a columnar data format that is easy and fast to version, query and train on.
It’s designed to be used with images, videos, 3D point clouds, audio and of course tabular data.
It supports any POSIX file systems, and cloud storage like AWS S3 and Google Cloud Storage.
The key features of Lance include:

* **High-performance random access:** 100x faster than Parquet.

* **Vector search:** find nearest neighbors in under 1 millisecond and combine OLAP-queries with vector search.

* **Zero-copy, automatic versioning:** manage versions of your data automatically, and reduce redundancy with zero-copy logic built-in.

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
   :maxdepth: 1

   Quickstart <./notebooks/quickstart>
   ./read_and_write
   File Format <./format>
   Arrays <./arrays>
   Integrations <./integrations/integrations>
   API References <./api/api>
   Contributor Guide <./contributing>
   Examples <./examples/examples>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
