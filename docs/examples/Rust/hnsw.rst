Indexing a dataset with HNSW(Hierarchical Navigable Small World)
-----------------------------------------

HNSW is a graph based algorithm for approximate neighbor search in high-dimensional spaces. In this example, we will demonstrate how to build an HNSW vector index against a Lance dataset. 

This example will show how to:

1. Generate synthetic test data of specified dimensions
2. Build a hierarchical graph structure for efficient vector search using Lance API
3. Perform vector search with different parameters and compute the ground truth using L2 distance search

.. literalinclude:: ../../../rust/examples/src/hnsw.rs
   :language: rust
   :linenos: