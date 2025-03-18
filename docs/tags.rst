Manage Tags
===========

Lance, much like Git, employs the :py:attr:`LanceDataset.tags <lance.LanceDataset.tags>`
property to label specific versions within a dataset's history.

:py:class:`Tags <lance.dataset.Tags>` are particularly useful for tracking the evolution of datasets,
especially in machine learning workflows where datasets are frequently updated.
For example, you can :py:meth:`create <lance.dataset.Tags.create>`, :meth:`update <lance.dataset.Tags.update>`,
and :meth:`delete <lance.dataset.Tags.delete>` or :py:meth:`list <lance.dataset.Tags.list>` tags.


.. testsetup::

    shutil.rmtree("./tags.lance", ignore_errors=True)
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    lance.write_dataset(data, "./tags.lance")
    data = [{"a": 5, "b": 6}, {"a": 7, "b": 8}]
    lance.write_dataset(data, "./tags.lance", mode="append")

.. doctest::

    >>> import lance
    >>> ds = lance.dataset("./tags.lance")
    >>> len(ds.versions())
    2
    >>> ds.tags.list()
    {}
    >>> ds.tags.create("v1-prod", 1)
    >>> ds.tags.list()
    {'v1-prod': {'version': 1, ...}}
    >>> ds.tags.update("v1-prod", 2)
    >>> ds.tags.list()
    {'v1-prod': {'version': 2, ...}}
    >>> ds.tags.delete("v1-prod")
    >>> ds.tags.list()
    {}

