# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Embedding vector utilities"""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Optional, Union

import pyarrow as pa
from tqdm.auto import tqdm

from . import write_dataset
from .dependencies import _check_for_numpy, torch
from .dependencies import numpy as np

if TYPE_CHECKING:
    from . import LanceDataset


def _normalize_vectors(vectors, ndim):
    if ndim is None:
        ndim = len(next(iter(vectors)))
    values = np.array(vectors, dtype="float32").ravel()
    return pa.FixedSizeListArray.from_arrays(values, list_size=ndim)


def _validate_ndim(values, ndim):
    for v in values:
        if ndim is None:
            ndim = len(v)
        else:
            if ndim != len(v):
                raise ValueError(f"Expected {ndim} dimensions but got {len(v)} for {v}")
    return ndim


def vec_to_table(
    data: Union[dict, list, np.ndarray],
    names: Optional[Union[str, list]] = None,
    ndim: Optional[int] = None,
    check_ndim: bool = True,
) -> pa.Table:
    """
    Create a pyarrow Table containing vectors.
    Vectors are created as FixedSizeListArray's in pyarrow with Float32 values.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> from lance.vector import vec_to_table
    >>> dd = {"vector0": np.random.randn(10), "vector1": np.random.randn(10)}
    >>> vec_to_table(dd)
    pyarrow.Table
    id: string
    vector: fixed_size_list<item: float>[10]
      child 0, item: float
    ----
    id: [["vector0","vector1"]]
    vector: [[[1.7640524,0.4001572,0.978738,2.2408931,1.867558,-0.9772779,0.95008844,\
-0.1513572,-0.10321885,0.41059852],[0.14404356,1.4542735,0.7610377,\
0.121675014,0.44386324,0.33367434,1.4940791,-0.20515826,0.3130677,-0.85409576]]]
    >>> vec_to_table(dd).to_pandas()
            id                                             vector
    0  vector0  [1.7640524, 0.4001572, 0.978738, 2.2408931, 1....
    1  vector1  [0.14404356, 1.4542735, 0.7610377, 0.121675014...

    Parameters
    ----------
    data: dict, list, or np.ndarray
        If dict, the keys are added as "id" column
        If list, then each element is assumed to be a vector
        If ndarray, then each row is assumed to be a vector
    names: str or list, optional
        If data is dict, then names should be a list of 2 str; default ["id", "vector"]
        If data is list or ndarray, then names should be str; default "vector"
    ndim: int, optional
        Number of dimensions of the vectors. Inferred if omitted.
    check_ndim: bool, default True
        Whether to verify that all vectors have the same length

    Returns
    -------
    tbl: pa.Table
        A pyarrow Table with vectors converted to appropriate types
    """
    if isinstance(data, dict):
        if names is None:
            names = ["id", "vector"]
        elif not isinstance(names, (list, tuple)) and len(names) == 2:
            raise ValueError(
                "If data is a dict, names must be a list or tuple of 2 strings"
            )
        values = list(data.values())
        if check_ndim:
            ndim = _validate_ndim(values, ndim)
        vectors = _normalize_vectors(values, ndim)
        ids = pa.array(data.keys())
        arrays = [ids, vectors]
    elif isinstance(data, list) or (
        _check_for_numpy(data) and isinstance(data, np.ndarray)
    ):
        if names is None:
            names = ["vector"]
        elif isinstance(names, str):
            names = [names]
        elif not isinstance(names, (list, tuple)) and len(names) == 1:
            raise ValueError(f"names cannot be more than 1 got {len(names)}")
        if check_ndim:
            ndim = _validate_ndim(data, ndim)
        vectors = _normalize_vectors(data, ndim)
        arrays = [vectors]
    else:
        raise NotImplementedError(
            f"data must be dict, list, or ndarray (require numpy installed), \
            got {type(data)} instead"
        )
    return pa.Table.from_arrays(arrays, names=names)


CUDA_REGEX = re.compile(r"^cuda(:\d+)?$")


def train_ivf_centroids_on_accelerator(
    dataset: LanceDataset,
    column: str,
    k: int,
    metric_type: Literal["l2", "cosine", "dot"],
    accelerator: Union[str, "torch.Device"],
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
) -> (np.ndarray, str):
    """Use accelerator (GPU or MPS) to train kmeans."""
    if isinstance(accelerator, str) and (
        not (CUDA_REGEX.match(accelerator) or accelerator == "mps")
    ):
        raise ValueError(
            "Train ivf centroids on accelerator: "
            + f"only support 'cuda' or 'mps' as accelerator, got '{accelerator}'."
        )

    sample_size = k * sample_rate

    from lance.torch.data import LanceDataset as TorchDataset

    from .torch.kmeans import KMeans

    k = int(k)

    logging.info("Randomly select %s centroids from %s", k, dataset)
    samples = dataset.sample(k, [column], sorted=True).combine_chunks()
    fsl = samples.to_batches()[0][column]
    init_centroids = torch.from_numpy(np.stack(fsl.to_numpy(zero_copy_only=False)))
    logging.info("Done sampling: centroids shape: %s", init_centroids.shape)

    ds = TorchDataset(
        dataset,
        batch_size=20480,
        columns=[column],
        samples=sample_size,
        cache=True,
    )

    logging.info("Training IVF partitions using GPU(%s)", accelerator)
    kmeans = KMeans(
        k,
        max_iters=max_iters,
        metric=metric_type,
        device=accelerator,
        centroids=init_centroids,
    )
    kmeans.fit(ds)

    centroids = kmeans.centroids.cpu().numpy()

    with tempfile.NamedTemporaryFile(delete=False) as f:
        np.save(f, centroids)
    logging.info("Saved centroids to %s", f.name)

    return centroids, compute_partitions(dataset, column, kmeans, batch_size=20480)


def compute_partitions(
    dataset: LanceDataset,
    column: str,
    kmeans: Any,  # KMeans
    batch_size: int = 10240,
    spill_dir: Union[str, Path] = None,
) -> str:
    """Compute partitions for each row using GPU kmeans and spill to disk.

    Parameters
    ----------
    dataset: LanceDataset
        Dataset to compute partitions for.
    column: str
        Column name of the vector column.
    kmeans: lance.torch.kmeans.KMeans
        KMeans model to use to compute partitions.
    batch_size: int, default 10240
        The batch size used to read the dataset.
    spill_dir: Path
        The path to store the partitions.

    Returns
    -------
    str
        The absolute path of the partition dataset.
    """
    from lance.torch.data import LanceDataset as PytorchLanceDataset

    torch_ds = PytorchLanceDataset(
        dataset,
        batch_size=batch_size,
        with_row_id=True,
        columns=[column],
    )
    output_schema = pa.schema([
        pa.field("row_id", pa.uint64()),
        pa.field("partition", pa.uint32()),
    ])

    def _partition_assignment() -> Iterable[pa.RecordBatch]:
        with torch.no_grad():
            for batch in torch_ds:
                batch: Dict[str, torch.Tensor] = batch
                vecs = batch[column].reshape(-1, kmeans.centroids.shape[1])

                vecs.to(kmeans.device)
                partitions = kmeans.transform(vecs).cpu().numpy()
                ids = batch["_rowid"].reshape(-1).cpu().numpy()

                # this is expected to be true, so just assert
                assert vecs.shape[0] == ids.shape[0]

                # Ignore any invalid vectors.
                ids = ids[partitions >= 0]
                partitions = partitions[partitions >= 0]
                part_batch = pa.RecordBatch.from_arrays(
                    [ids, partitions],
                    schema=output_schema,
                )
                if len(part_batch) < len(ids):
                    logging.warning(
                        "%s vectors are ignored during partition assignment",
                        len(part_batch) - len(ids),
                    )

                yield part_batch

    rbr = pa.RecordBatchReader.from_batches(
        output_schema, tqdm(_partition_assignment())
    )

    if spill_dir is None:
        spill_dir = tempfile.mkdtemp()

    spill_uri = Path(spill_dir) / "precomputed_partitions.lance"

    ds = write_dataset(
        rbr,
        spill_uri,
        schema=output_schema,
        max_rows_per_file=dataset.count_rows(),
    )
    assert len(ds.get_fragments()) == 1
    files = ds.get_fragments()[0].data_files()
    assert len(files) == 1

    return str(spill_uri)
