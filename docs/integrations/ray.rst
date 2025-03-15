Lance ❤️ Ray
--------------------

`Ray <https://www.anyscale.com/product/open-source/ray>`_ effortlessly scale up ML workload to large distributed
compute environment.

Lance format is one of the official `Ray data sources <https://docs.ray.io/en/latest/data/api/input_output.html#lance>`_:

* Lance Data Source :py:meth:`ray.data.read_lance`
* Lance Data Sink :py:meth:`ray.data.Dataste.write_lance`

.. testsetup:: ray

    shutil.rmtree("./alice_bob_and_charlie.lance", ignore_errors=True)

.. testcode:: ray

    import ray

    ray.init()

    data = [
        {"age": 25, "name": "alice"},
        {"age": 33, "name": "bob"},
        {"age": 44, "name": "charlie"}
    ]
    ray.data.from_items(data).write_lance("./alice_bob_and_charlie.lance")

    # It can be read via lance directly
    pdf = lance.dataset("./alice_bob_and_charlie.lance").to_table().to_pandas()
    assert pdf.equals(pd.DataFrame(data))

    # Or via Ray.data.read_lance
    pd_df = ray.data.read_lance("./alice_bob_and_charlie.lance").to_pandas()
    assert pdf.equals(pd_df)