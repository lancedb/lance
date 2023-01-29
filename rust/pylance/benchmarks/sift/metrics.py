import duckdb
import lance
import numpy as np


def recall(actual_sorted: np.ndarray, results: np.ndarray):
    """
    Recall-at-k
    """
    len = results.shape[1]
    t = actual_sorted[:, len - 1] + 1e-3
    recall_at_k = (results <= t[:, None]).sum(axis=1) * 1.0 / len
    return (recall_at_k.mean(),
            recall_at_k.std(),
            recall_at_k)


def l2_sort(mat, q):
    """
    Compute the actual euclidean squared

    Parameters
    ----------
    mat: ndarray
        shape is (n, d) where n is number of vectors and d is number of dims
    q: ndarray
        shape is d, this is the query vector
    """
    return np.sort(((mat - q)**2).sum(axis=1))


def l2_part(mat, q, k):
    """
    Compute topk by partition

    Parameters
    ----------
    mat: ndarray
        shape is (n, d) where n is number of vectors and d is number of dims
    q: ndarray
        shape is d, this is the query vector
    k: int
        topk
    """
    return np.partition(((mat - q)**2).sum(axis=1), k)[:k]


def test(nsamples=100):
    """
    make sure the recall computation is correct.
    if we just use np.partition, we should have perfect recall
    """
    mat = np.random.randn(1000000, 128)
    mat = mat / np.sqrt((mat**2).sum(axis=1))[:, None]  # to unit vectors
    actual_sorted = []
    results = []
    for _ in range(nsamples):
        q = mat[np.random.randint(mat.shape[0]), :]
        actual_sorted.append(l2_sort(mat, q))
        results.append(l2_part(mat, q, 10))
    rs = recall(np.array(actual_sorted), np.array(results))
    assert np.abs(rs.mean() - 1.0) < 1e-3


def test_dataset(uri, nsamples=100, k=10):
    dataset = lance.dataset(uri)
    tbl = dataset.to_table()
    v = tbl["vector"].combine_chunks()
    all_vectors = v.values.to_numpy().reshape(len(tbl), v.type.list_size)

    query_vectors = duckdb.query(f"SELECT vector FROM tbl USING SAMPLE {nsamples}").to_df()
    query_vectors = np.array([np.array(x) for x in query_vectors.vector.values])

    actual_sorted = []
    results = []

    for i in range(nsamples):
        q = query_vectors[i, :]
        actual_sorted.append(l2_sort(all_vectors, q))
        results.append(
            dataset.to_table(
                nearest={"column": "vector", "q": q, "k": k}
            )["score"].combine_chunks().to_numpy()
        )
    return recall(np.array(actual_sorted), np.array(results))

