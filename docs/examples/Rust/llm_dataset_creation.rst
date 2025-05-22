Creating text dataset for LLM training using Lance in Rust
----------------------------------------------------------

In this example, we will demonstrate how to achieve the Python example - LLM dataset creation shown in :doc:`../Python/llm_dataset_creation` in Rust.


.. note::
   The huggingface Python API supports loading data in streaming mode and shuffling is provided as a builtin feature. Rust API lacks these feature thus the data are manually downloaded and shuffled within each batch.

This example will show how to:

1. Download and process a text dataset in parts from huggingface
2. Tokenize the text data with a custom RecordBatchReader
3. Save it as a Lance dataset using Lance API

The implementation details in Rust will follow similar concepts as the Python version, but with Rust-specific APIs and patterns which are significantly more verbose.

.. literalinclude:: ../../../rust/examples/src/llm_dataset_creation.rs
   :language: rust
   :linenos: