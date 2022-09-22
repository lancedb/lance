Lance: A Columnar Data Format for Computer Vision
=================================================

`Lance` is a cloud-native columnar data format designed for
managing large-scale computer vision datasets in production environments.
Lance delivers blazing fast performance for image and video data use cases from analytics
to point queries to training scans.


What problems does Lance solve?
-------------------------------

Today, the data tooling stack for computer vision is insufficient to serve
the needs of the ML engineering community.

Working with vision data for ML is different from working with tabular data:

* Training, analytics, and labeling uses different tools requiring different formats
* Data annotations are almost always deeply nested
* Images / videos are large blobs that are difficult to query by existing engines

This results in some major pain-points:

* Too much time spent on low level data munging
* Multiple copies creates data quality issues, even for well-known datasets
* Reproducibility and data versioning is extremely difficult to achieve

Lance to the rescue
-------------------
To solve these pain-points, we are building Lance, an open-source columnar data format optimized for computer vision with the following goals:

* Blazing fast performance for analytical scans and random access to individual records (for visualization and annotation)
* Rich ML data types and integrations to eliminate manual data conversions.
* Support for vector and search indices, versioning, and schema evolution.



.. toctree::
   :maxdepth: 2

   quickstart
   API References <./api/api>



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
