Python APIs
-----------

``Lance`` is a columnar format that is specifically designed for efficient
multi-modal data processing.

Lance Dataset
~~~~~~~~~~~~~

The core of Lance is the ``LanceDataset`` class. User can open a dataset by using
:py:meth:`lance.dataset`.

.. autofunction:: lance.dataset

Basic IOs
~~~~~~~~~

The following functions are used to read and write data in Lance format.

.. automethod:: lance.dataset.LanceDataset.insert
.. automethod:: lance.dataset.LanceDataset.scanner
.. automethod:: lance.dataset.LanceDataset.to_batches

Random Access
~~~~~~~~~~~~~

Lance stands out with its super fast random access, unlike other columnar formats.

.. automethod:: lance.dataset.LanceDataset.take

.. automethod:: lance.dataset.LanceDataset.take_blobs


Schema Evolution
~~~~~~~~~~~~~~~~


API Reference
~~~~~~~~~~~~~

More information can be found in the :doc:`API reference <python/modules>`.

.. _Lance Python API documentation: ./python/modules
