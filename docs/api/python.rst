Python APIs
===========

``Lance`` is a columnar format that is specifically designed for efficient
multi-modal data processing.

Lance Dataset
-------------

The core of Lance is the ``LanceDataset`` class. User can open a dataset by using
:py:meth:`lance.dataset`.

.. autofunction:: lance.dataset
    :noindex:

Basic IOs
~~~~~~~~~

The following functions are used to read and write data in Lance format.

.. automethod:: lance.dataset.LanceDataset.insert
    :noindex:
.. automethod:: lance.dataset.LanceDataset.scanner
    :noindex:
.. automethod:: lance.dataset.LanceDataset.to_batches
    :noindex:
.. automethod:: lance.dataset.LanceDataset.to_table
    :noindex:

Random Access
~~~~~~~~~~~~~

Lance stands out with its super fast random access, unlike other columnar formats.

.. automethod:: lance.dataset.LanceDataset.take
    :noindex:
.. automethod:: lance.dataset.LanceDataset.take_blobs
    :noindex:


Schema Evolution
~~~~~~~~~~~~~~~~

Lance supports schema evolution, which means that you can add new columns to the dataset
cheaply.

.. automethod:: lance.dataset.LanceDataset.add_columns
    :noindex:
.. automethod:: lance.dataset.LanceDataset.drop_columns
    :noindex:


Indexing and Searching
~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: lance.dataset.LanceDataset.create_index
    :noindex:
.. automethod:: lance.dataset.LanceDataset.scanner
    :noindex:

API Reference
~~~~~~~~~~~~~

More information can be found in the :doc:`API reference <python/modules>`.

.. _Lance Python API documentation: ./python/modules
