Lance ❤️ Ray
====================

`Ray <https://www.anyscale.com/product/open-source/ray>`_ effortlessly scale up ML workload to large distributed
compute environment.


Basic Operations
--------------------

Lance format is one of the official `Ray data sources <https://docs.ray.io/en/latest/data/api/input_output.html#lance>`_:

* Lance Data Source :py:meth:`ray.data.read_lance`
* Lance Data Sink :py:meth:`ray.data.Dataste.write_lance`

.. testsetup::

    shutil.rmtree("./alice_bob_and_charlie.lance", ignore_errors=True)

.. testcode::

    import ray
    import pandas as pd

    ray.init()

    data = [
        {"id": 1, "name": "alice"},
        {"id": 2, "name": "bob"},
        {"id": 3, "name": "charlie"}
    ]
    ray.data.from_items(data).write_lance("./alice_bob_and_charlie.lance")

    # It can be read via lance directly
    df = (
        lance.
        dataset("./alice_bob_and_charlie.lance")
        .to_table()
        .to_pandas()
        .sort_values(by=["id"])
        .reset_index(drop=True)
    )
    assert df.equals(pd.DataFrame(data)), "{} != {}".format(
        df, pd.DataFrame(data)
    )

    # Or via Ray.data.read_lance
    ray_df = (
        ray.data.read_lance("./alice_bob_and_charlie.lance")
        .to_pandas()
        .sort_values(by=["id"])
        .reset_index(drop=True)
    )
    assert df.equals(ray_df)

Advanced Operations
--------------------

Parallel Column Merging
^^^^^^^^^^^^^^^^^^^^^^

Demonstration of parallel column generation using Lance's native operations:

.. code-block:: python

    import pyarrow as pa
    from pathlib import Path
    import lance

    # Define schema
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("height", pa.int64()),
        pa.field("weight", pa.int64()),
    ])

    # Generate initial dataset
    ds = (
        ray.data.range(10)  # Create 0-9 IDs
        .map(lambda x: {
            "id": x["id"],
            "height": x["id"] + 5,  # height = id + 5
            "weight": x["id"] * 2   # weight = id * 2
        })
        .write_lance(str(output_path), schema=schema)
    )

    # Define label generation logic
    def generate_labels(batch: pa.RecordBatch) -> pa.RecordBatch:
        heights = batch.column("height").to_pylist()
        size_labels = ["tall" if h > 8 else "medium" if h > 6 else "short" for h in heights]
        return pa.RecordBatch.from_arrays([
            pa.array(size_labels)
        ], names=["size_labels"])

    # Add new columns in parallel
    lance_ds = lance.dataset(output_path)
    add_columns(
        lance_ds,
        generate_labels,
        source_columns=["height"],  # Input columns needed
    )

    # Display final results
    final_df = lance_ds.to_table().to_pandas()
    print("\\nEnhanced dataset with size labels:\\n")
    print(final_df.sort_values("id").to_string(index=False))
