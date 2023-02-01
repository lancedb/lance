Lance: Blazing fast exploration and analysis of machine learning visual data using SQL
======================================================================================

Lance makes machine learning workflows with visual data easy (images, videos, point clouds, audio, and more), by allowing Developers, Analysts and Operations to:

* Use arbitary ML functions in SQL for common use cases such as similarity search using embeddings, model inference and computing evaluation metrics like F1, IOU and more.

* [Coming soon] Visualize, slice and drill-into visual datasets to inspect embeddings, labels/annotations, metrics and more.

* [Coming soon] Version, compare and diff visual datasets easily.

Lance is powered by Lance Format, an Apache-Arrow compatible columnar data format which is an alternative to Parquet, Iceberg and Delta. Lance has 50-100x faster query performance for visual data use cases.

Lance currently supports DuckDB.

.. toctree::
   :maxdepth: 1

   Quickstart <./notebooks/quickstart>
   howtos
   File Format <./format>
   API References <./api/api>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
