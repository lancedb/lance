from typing import Literal
import numpy as np
import pyarrow as pa
import lance
from pathlib import Path
from typing import List

import gzip
import requests


from scipy.sparse import lil_matrix
from sklearn import random_projection
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm.auto import tqdm


def l2(X, Y):
    sx = np.sum(X**2, axis=1, keepdims=True)
    sy = np.sum(Y**2, axis=1, keepdims=True)
    return -2 * X.dot(Y.T) + sx + sy.T


def cosine(X, Y):
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    return 1 - (X @ Y.T)


def knn(
    query: np.ndarray,
    data: np.ndarray,
    metric: Literal["L2", "cosine"],
    k: int,
) -> np.ndarray:
    if metric == "L2":
        dist = l2
    elif metric == "cosine":
        dist = cosine
    else:
        raise ValueError("Invalid metric")
    return np.argpartition(dist(query, data), k, axis=1)[:, 0:k]


def write_lance(
    path: str,
    data: np.ndarray,
):
    dims = data.shape[1]

    schema = pa.schema(
        [
            pa.field("vec", pa.list_(pa.float32(), dims)),
            pa.field("id", pa.uint32(), False),
        ]
    )

    fsl = pa.FixedSizeListArray.from_arrays(
        pa.array(data.reshape(-1).astype(np.float32), type=pa.float32()),
        dims,
    )
    ids = pa.array(range(data.shape[0]), type=pa.uint32())
    t = pa.Table.from_pydict({"vec": fsl, "id": ids}, schema)

    lance.write_dataset(t, path)


# NYT

_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz"


def _get_nyt_data():
    print("starting download")
    with requests.get(_DATA_URL) as resp:
        print("download complete, decompressing...")
        return gzip.decompress(resp.content).decode().split("\n")


_CACHE_PATH = Path("/tmp/nyt_data.npy")


# rewrite of https://github.com/erikbern/ann-benchmarks/blob/b64fcb98172b0aa133f656b0e0e8d49d481e0896/ann_benchmarks/datasets.py#L344
def _get_nyt_vectors(
    data: List[str] = None,
    output_dims: int = 256,
) -> np.ndarray:
    if _CACHE_PATH.exists():
        print("loading from cache")
        return np.load(_CACHE_PATH)
    if data is None:
        data = _get_nyt_data()
    # the file is formatted as
    #
    # num entries
    # num words
    # (doc_id, word_id, count)
    # (doc_id, word_id, count)
    # ...
    #
    # 1-indexed
    entries = int(data[0])
    words = int(data[1])

    freq = lil_matrix((entries, words))
    for e in tqdm(data[3:], desc="populating freq matrix"):
        if e == "":
            continue
        doc, word, cnt = [int(v) for v in e.strip().split()]
        freq[doc - 1, word - 1] = cnt
    print("computing tfidf")
    tfidf = TfidfTransformer().fit_transform(freq)
    print("computing dense projection")
    dense_projection = random_projection.GaussianRandomProjection(
        n_components=output_dims,
        random_state=42,
    ).fit_transform(tfidf)
    dense_projection = dense_projection.astype(np.float32)
    np.save(_CACHE_PATH, dense_projection)
    return dense_projection
