Data Evolution
==============

Lance supports zero-copy data evolution, which means that you can add new columns and
backfill column data to the dataset cheaply.

.. automethod:: lance.dataset.LanceDataset.add_columns
    :noindex:
.. automethod:: lance.dataset.LanceDataset.drop_columns
    :noindex: