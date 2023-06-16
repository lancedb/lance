Vector search with indices
==========================

Lance supports queries with Approximate Nearest Neighbors (ANN) algorithms. This
gives you answers to nearest neighbor queries very quickly, but one must be
careful in how we use it to get good results. 

A mental model of the query pipeline
------------------------------------

To understand the query behavior, we'll look at an example query:

.. code-block:: python

    dataset.to_table(
        nearest={
            "column": "vector", 
            "q": [0.1, 0.2, 0.3],
            "k": 10, 
            "nprobes": 10, 
            "refine_factor": 5
        },
        filter="c in (1, 2, 3)",
        limit=10,
        offset=10,
        columns=['a', 'b'],
    )

``nprobes``: what does this mean for query consistency?

How does this query run?

1. The ``nearest`` parameter is used to start a search for nearest neighbors.
2. The ``filter`` parameter is used to prune the nearest neighbor search results
   until we have found ``k * refine_factor`` neighbors. So here, we find the top
   50 candidates. In order to perform the filter, we retrieve the ``c`` column
   for every search result we get from the previous step.
3. We refine the results by getting their actual vector values, computing the
   exact distance, and then taking the ``k`` nearest neighbors. This step
   improves the accuracy of vector search results substantially.
   * In this step, we also merge in flat search results from rows that aren't
     part of the index.
4. We apply the limit and offset to the list, skipping ``offset`` values and then
   taking ``limit`` values.
5. We retrieve the columns ``a`` and ``b`` for the final results.

.. note::

    In previous versions, ``k`` referred to the number of neighbors **before**
    filtering. Since version 0.5.0, ``k`` refers to the number of neighbors
    **after** filtering.


Paginating vector search results
--------------------------------

If you want to paginate results, it's important to understand the guarantees on
these queries.

A given nearest query + filter combination will always return the same results.
But if you change any of the parameters, the results may change.

So if you want to create paginated results, with a page size of 10, you can set
``limit=10``, and then ``k`` to the maximum number of results you will ever
retrieve. So for up to 10 pages, ``k=100``. For up to 100 pages, ``k=1000``. The
higher the number, the higher the ``k``, the higher the latency.