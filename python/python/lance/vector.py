# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Embedding vector utilities"""

from __future__ import annotations

import logging
import re
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Optional, Union

import pyarrow as pa
from tqdm.auto import tqdm

import lance

from . import write_dataset
from .dependencies import _check_for_numpy, torch
from .dependencies import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

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


def train_pq_codebook_on_accelerator(
    dataset: LanceDataset,
    metric_type: Literal["l2", "cosine", "dot"],
    kmeans: Any,
    accelerator: Union[str, "torch.Device"],
    num_sub_vectors: int,
    batch_size: int = 1024 * 10 * 4,
) -> (np.ndarray, List[Any]):
    """Use accelerator (GPU or MPS) to train pq codebook."""
    from lance.torch.data import _to_tensor as to_full_tensor

    # TODO residual vec can be split up into mulitple fields _during_ ivf assignment
    column = "__residual_vec"

    # cuvs not particularly useful for only 256 centroids without more work
    if accelerator == "cuvs":
        accelerator = "cuda"

    device = kmeans.device

    centroids_list = []
    kmeans_list = []

    field_names = [f"{column}_{i + 1}" for i in range(num_sub_vectors)]
    dim = kmeans.centroids.shape[1]
    subvector_size = dim // num_sub_vectors
    fields = [pa.field(name, pa.list_(pa.float32(), list_size=subvector_size)) for name in field_names]
    split_schema = pa.schema(fields)

    from lance.torch.data import LanceDataset as PytorchLanceDataset
    torch_ds = PytorchLanceDataset(
       dataset,
       batch_size=batch_size,
       with_row_id=False,
       columns=[column],
    )
    loader = torch.utils.data.DataLoader(
       torch_ds,
       batch_size=1,
       pin_memory=True,
       collate_fn=_collate_fn,
    )
    def split_batches() -> Iterable[pa.RecordBatch]:
        with torch.no_grad():
            for batch in loader:
                #batch = to_full_tensor(batch)
                split_columns = []
                vector_dim = batch.size(1)
                #subvector_size = vector_dim // num_sub_vectors
                for i in range(num_sub_vectors):
                    subvector_tensor = batch[:, i * subvector_size: (i + 1) * subvector_size]
                    subvector_arr = pa.array(subvector_tensor.cpu().detach().numpy().reshape(-1))
                    subvector_fsl = pa.FixedSizeListArray.from_arrays(subvector_arr, subvector_size)
                    #subvector_array = pa.FixedShapeTensorArray.from_numpy_ndarray(subvector_tensor.cpu().detach().numpy())
                    split_columns.append(subvector_fsl)
                new_batch = pa.RecordBatch.from_arrays(split_columns, schema=split_schema)
                yield new_batch

    rbr = pa.RecordBatchReader.from_batches(split_schema, split_batches())
    split_dataset_uri = tempfile.mkdtemp()
    ds_split = write_dataset(
        rbr,
        split_dataset_uri,
        schema=split_schema,
        max_rows_per_file=dataset.count_rows(),
        data_storage_version="stable",
    )

    for sub_vector in range(num_sub_vectors):
        ivf_centroids_local, kmeans_local = train_ivf_centroids_on_accelerator(
            ds_split,
            #dataset,
            #"__residual_vec",
            field_names[sub_vector],
            256,
            metric_type,
            accelerator,
        )
        centroids_list.append(ivf_centroids_local)
        kmeans_list.append(kmeans_local)

    pq_codebook = np.stack(centroids_list)
    return pq_codebook, kmeans_list


def train_ivf_centroids_on_accelerator(
    dataset: LanceDataset,
    column: str,
    k: int,
    metric_type: Literal["l2", "cosine", "dot"],
    accelerator: Union[str, "torch.Device"],
    batch_size: int = 1024 * 10 * 4,
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
) -> (np.ndarray, Any):
    """Use accelerator (GPU or MPS) to train kmeans."""
    if isinstance(accelerator, str) and (
        not (
            CUDA_REGEX.match(accelerator)
            or accelerator == "mps"
            or accelerator == "cuvs"
        )
    ):
        raise ValueError(
            "Train ivf centroids on accelerator: "
            + f"only support 'cuda' or 'mps' as accelerator, got '{accelerator}'."
        )

    sample_size = k * sample_rate

    from lance.torch.data import LanceDataset as TorchDataset

    from .torch.kmeans import KMeans

    k = int(k)

    if dataset.schema.field(column).nullable:
        filt = f"{column} is not null"
    else:
        filt = None

    logging.info("Randomly select %s centroids from %s (filt=%s)", k, dataset, filt)

    ds = TorchDataset(
        dataset,
        batch_size=k,
        columns=[column],
        samples=sample_size,
        filter=filt,
    )

    init_centroids = next(iter(ds))

    logging.info("Done sampling: centroids shape: %s", init_centroids.shape)

    ds = TorchDataset(
        dataset,
        batch_size=20480,
        columns=[column],
        samples=sample_size,
        #filter=filt,
        cache=True,
    )

    if accelerator == "cuvs":
        logging.info("Training IVF partitions using cuVS+GPU")
        print("Training IVF partitions using cuVS+GPU")
        from lance.cuvs.kmeans import KMeans as KMeansCuVS

        kmeans = KMeansCuVS(
            k,
            max_iters=max_iters,
            metric=metric_type,
            device="cuda",
            centroids=init_centroids,
        )
    else:
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

    return centroids, kmeans


def compute_pq_codes(
    dataset: LanceDataset,
    kmeans_list: List[Any],  # KMeans
    batch_size: int = 1024 * 10 * 4,
    dst_dataset_uri: Optional[Union[str, Path]] = None,
    allow_cuda_tf32: bool = True,
) -> str:
    """Compute pq codes for each row using GPU kmeans and spill to disk.

    Parameters
    ----------
    dataset: LanceDataset
        Dataset to compute pq codes for.
    kmeans_list: List[lance.torch.kmeans.KMeans]
        KMeans models to use to compute pq (one per subspace)
    batch_size: int, default 10240
        The batch size used to read the dataset.
    dst_dataset_uri: Union[str, Path], optional
        The path to store the partitions.  If not specified a random
        directory is used instead
    allow_tf32: bool, default True
        Whether to allow tf32 for matmul on CUDA.

    Returns
    -------
    str
        The absolute path of the pq codes dataset.
    """

    torch.backends.cuda.matmul.allow_tf32 = allow_cuda_tf32

    num_rows = dataset.count_rows()

    num_sub_vectors = len(kmeans_list)

    from lance.torch.data import LanceDataset as PytorchLanceDataset
    torch_ds = PytorchLanceDataset(
       dataset,
       batch_size=batch_size,
       with_row_id=False,
       #columns=[column, "__residual_vec"], #, "id", "partition"],
       columns=["_rowid", "__ivf_part_id", "__residual_vec"],
    )
    loader = torch.utils.data.DataLoader(
       torch_ds,
       batch_size=1,  # TODO is this significantly inhibiting performance
       pin_memory=True,
       collate_fn=_collate_fn,
    )
    output_schema = pa.schema(
        [
            pa.field("_rowid", pa.uint64()),
            pa.field("__ivf_part_id", pa.uint32()),
            pa.field("__pq_code", pa.list_(pa.uint8(), list_size=num_sub_vectors)),
        ]
    )

    progress = tqdm(total=num_rows)

    device = kmeans_list[0].device

    from lance.torch.data import _to_tensor as to_full_tensor

    def _pq_codes_assignment() -> Iterable[pa.RecordBatch]:
        with torch.no_grad():
            for batch in loader:
                #batch = to_full_tensor(batch)
                # batch["__residual_vec"]
                vecs = batch["__residual_vec"].to(device).reshape(
                    -1, kmeans_list[0].centroids.shape[1] * len(kmeans_list)
                )

                ids = batch["_rowid"].reshape(-1)
                partitions = batch["__ivf_part_id"].reshape(-1)

                sub_vecs = vecs.view(
                    vecs.shape[0], num_sub_vectors, vecs.shape[1] // num_sub_vectors
                )
                pq_codes = torch.stack(
                    [
                        kmeans_list[i].transform(sub_vecs[:, i, :])
                        for i in range(num_sub_vectors)
                    ],
                    dim=1,
                )
                pq_codes = pq_codes.to(torch.uint8)

                ids = ids.cpu()
                partitions = partitions.cpu()
                pq_codes = pq_codes.cpu()

                pq_values = pa.array(pq_codes.numpy().reshape(-1))
                pq_codes = pa.FixedSizeListArray.from_arrays(pq_values, num_sub_vectors)
                part_batch = pa.RecordBatch.from_arrays(
                    [ids, partitions, pq_codes],
                    schema=output_schema,
                )

                progress.update(part_batch.num_rows)
                yield part_batch

    rbr = pa.RecordBatchReader.from_batches(output_schema, _pq_codes_assignment())
    if dst_dataset_uri is None:
       dst_dataset_uri = tempfile.mkdtemp()
    ds = write_dataset(
       rbr,
       dst_dataset_uri,
       schema=output_schema,
       max_rows_per_file=dataset.count_rows(),
       data_storage_version="stable",
    )
    assert len(ds.get_fragments()) == 1
    files = ds.get_fragments()[0].data_files()
    assert len(files) == 1

    progress.close()

    logging.info("Saved precomputed pq_codes to %s", dst_dataset_uri)
    return dst_dataset_uri


def _collate_fn(batch):
    return batch[0]


def compute_partitions(
    dataset: LanceDataset,
    column: str,
    kmeans: Any,  # KMeans
    batch_size: int = 1024 * 10 * 4,
    dst_dataset_uri: Optional[Union[str, Path]] = None,
    allow_cuda_tf32: bool = True,
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
    dst_dataset_uri: Union[str, Path], optional
        The path to store the partitions.  If not specified a random
        directory is used instead
    allow_tf32: bool, default True
        Whether to allow tf32 for matmul on CUDA.

    Returns
    -------
    str
        The absolute path of the partition dataset.
    """
    from lance.torch.data import LanceDataset as PytorchLanceDataset

    torch.backends.cuda.matmul.allow_tf32 = allow_cuda_tf32

    num_rows = dataset.count_rows()

    torch_ds = PytorchLanceDataset(
        dataset,
        batch_size=batch_size,
        with_row_id=True,
        columns=[column],
        filter=f"{column} is not null",
    )
    loader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=1,
        pin_memory=True,
        collate_fn=_collate_fn,
    )

    dim = kmeans.centroids.shape[1]

    # TODO add option to not have residual vec. Also split into subvecs here.

    output_schema = pa.schema(
        [
            pa.field("_rowid", pa.uint64()),
            pa.field("__ivf_part_id", pa.uint32()),
            pa.field("__residual_vec", pa.list_(pa.float32(), list_size=dim)),
        ]
    )

    progress = tqdm(total=num_rows)

    def _partition_assignment() -> Iterable[pa.RecordBatch]:
        with torch.no_grad():
            for batch in loader:
                vecs = (
                    batch[column]
                    .to(kmeans.device)
                    .reshape(-1, kmeans.centroids.shape[1])
                )


                partitions = kmeans.transform(vecs)
                ids = batch["_rowid"].reshape(-1)
                # this is expected to be true, so just assert
                assert vecs.shape[0] == ids.shape[0]

                residual_vecs = vecs - kmeans.centroids[partitions]

                # Ignore any invalid vectors.
                mask = (partitions.isfinite()).cpu()
                ids = ids[mask]
                partitions = partitions.cpu()[mask]

                residual_vecs = residual_vecs.cpu()[mask]
                residual_vecs = residual_vecs.to(vecs.dtype)

                residual_values = pa.array(residual_vecs.numpy().reshape(-1))
                residual_vecs = pa.FixedSizeListArray.from_arrays(residual_values, dim)

                part_batch = pa.RecordBatch.from_arrays(
                    [
                        ids.numpy(),
                        partitions.numpy(),
                        residual_vecs,
                    ],
                    schema=output_schema,
                )
                if len(part_batch) < len(ids):
                    logging.warning(
                        "%s vectors are ignored during partition assignment",
                        len(part_batch) - len(ids),
                    )

                progress.update(part_batch.num_rows)
                yield part_batch

    rbr = pa.RecordBatchReader.from_batches(output_schema, _partition_assignment())
    if dst_dataset_uri is None:
        dst_dataset_uri = tempfile.mkdtemp()
    ds = write_dataset(
        rbr,
        dst_dataset_uri,
        schema=output_schema,
        max_rows_per_file=dataset.count_rows(),
        data_storage_version="stable",
    )
    assert len(ds.get_fragments()) == 1
    files = ds.get_fragments()[0].data_files()
    assert len(files) == 1

    progress.close()

    logging.info("Saved precomputed partitions to %s", dst_dataset_uri)
    return str(dst_dataset_uri)
