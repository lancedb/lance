HuggingFace
-----------

The Hugging Face Hub has a great amount of pre-trained models and datasets available.

It is easy to convert a Huggingface dataset to Lance dataset:

.. code-block:: python

    # Huggingface datasets
    import datasets
    import lance

    lance.write(datasets.load_dataset(
        "poloclub/diffusiondb", split="train[:10]"
    ), "diffusiondb_train.lance")