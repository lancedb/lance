Lance ❤️ Ray
--------------------

Ray effortlessly scale up ML workload to large distributed compute environment.

`Ray Data <https://docs.ray.io/en/latest/data/data.html>`_ can be directly written in Lance format by using the
:class:`lance.ray.sink.LanceDatasink` class. For example:

.. code-block:: bash

    pip install pylance[ray]


``Ray Data Dataset`` can be written to Lance format using the following code:

.. code-block:: python

    import ray
    from lance.ray.sink import LanceDatasink

    sink = LanceDatasink("s3://bucket/to/data.lance")
    ray.data.range(10).map(
        lambda x: {"id": x["id"], "str": f"str-{x['id']}"}
    ).write_datasink(sink)

