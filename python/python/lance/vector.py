# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""Embedding vector utilities"""

from __future__ import annotations

import logging
import re
import tempfile
from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Optional, Union

import lance

import pyarrow as pa
from tqdm.auto import tqdm

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


@torch.jit.script
def to_sub_vectors(
    full: torch.Tensor, num_sub_vectors: int, sub_vector: int
) -> torch.Tensor:
    dim = full.shape[1]
    assert (
        dim % num_sub_vectors == 0
    ), f"num_sub_vectors ({num_sub_vectors}) must divide dim ({dim})."
    sub_vector_size = dim // num_sub_vectors
    start_idx = sub_vector * sub_vector_size
    end_idx = start_idx + sub_vector_size
    return full[:, start_idx:end_idx]


def train_pq_codebook_on_accelerator(
    dataset: LanceDataset,
    column: str,
    metric_type: Literal["l2", "cosine", "dot"],
    kmeans: Any,
    accelerator: Union[str, "torch.Device"],
    num_sub_vectors: int,
) -> (np.ndarray, List[Any]):
    """Use accelerator (GPU or MPS) to train pq codebook."""
    from lance.torch.data import _to_tensor as to_full_tensor

    # cuvs not particularly useful for only 256 centroids without more work
    if accelerator == "cuvs":
        accelerator = "cuda"

    device = kmeans.device

    centroids_list = []
    kmeans_list = []

    # TODO make new temp dataset files instead of using a to_tensor override fn

    for sub_vector in range(num_sub_vectors):

        def modify_tensor_fn(
            full: torch.Tensor,
        ) -> torch.Tensor:
            full = full.to(device)
            return to_sub_vectors(full, num_sub_vectors, sub_vector)

        def to_tensor_local(
            batch: pa.RecordBatch,
            *,
            uint64_as_int64: bool = True,
            hf_converter: Optional[dict] = None,
        ) -> Union[dict[str, torch.Tensor], torch.Tensor]:
            full = to_full_tensor(
                batch, uint64_as_int64=uint64_as_int64, hf_converter=hf_converter
            )
            return modify_tensor_fn(full)

        ivf_centroids_local, kmeans_local = train_ivf_centroids_on_accelerator(
            dataset,
            "__residual_vec",
            # column,
            256,
            metric_type,
            accelerator,
            to_tensor_fn=to_tensor_local,
            modify_tensor_fn=modify_tensor_fn,
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
    *,
    sample_rate: int = 256,
    max_iters: int = 50,
    to_tensor_fn: Optional[
        callable[[pa.RecordBatch], Union[dict[str, torch.Tensor], torch.Tensor]]
    ] = None,
    modify_tensor_fn: Optional[callable[[torch.Tensor], torch.Tensor]] = None,
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

    if modify_tensor_fn is not None:
        init_centroids = modify_tensor_fn(init_centroids)

    logging.info("Done sampling: centroids shape: %s", init_centroids.shape)

    ds = TorchDataset(
        dataset,
        batch_size=20480,
        columns=[column],
        samples=sample_size,
        filter=filt,
        cache=True,
        to_tensor_fn=to_tensor_fn,
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


@torch.jit.script
def compute_pq_codes_batch(
    vecs: torch.Tensor,
    centroids: torch.Tensor,
    num_sub_vectors: int,
    sub_vector_size: int,
):
    # TODO support other distances, move this to a different file

    sub_vecs = vecs.view(
        vecs.shape[0], num_sub_vectors, sub_vector_size
    )  # Shape: (batch_size, num_sub_vectors, sub_vector_size)

    # Compute the L2 distance between each sub-vector and its corresponding centroids

    # (batch_size, num_sub_vectors, sub_vector_size) -> (batch_size, num_sub_vectors, 1, sub_vector_size)
    sub_vecs_expanded = sub_vecs.unsqueeze(2)

    # (num_sub_vectors, num_centroids, sub_vector_size) -> (1, num_sub_vectors, num_centroids, sub_vector_size)
    centroids_expanded = centroids.unsqueeze(0)

    # Compute squared L2 distance
    # Shape: (batch_size, num_sub_vectors, num_centroids)
    dists = torch.sum((sub_vecs_expanded - centroids_expanded) ** 2, dim=-1)

    # Get the index of the nearest centroid for each sub-vector (i.e., the PQ code)
    pq_codes = torch.argmin(dists, dim=-1)  # Shape: (batch_size, num_sub_vectors)

    return pq_codes


def compute_pq_codes(
    dataset: LanceDataset,
    column: str,
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
    column: str
        Column name of the vector column.
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
    from lance.torch.data import LanceDataset as PytorchLanceDataset

    torch.backends.cuda.matmul.allow_tf32 = allow_cuda_tf32

    num_rows = dataset.count_rows()

    num_sub_vectors = len(kmeans_list)

    # torch_ds = PytorchLanceDataset(
    #    dataset,
    #    batch_size=batch_size,
    #    with_row_id=True,
    #    columns=[column, "__residual_vec"], #, "id", "partition"],
    # )
    # loader = torch.utils.data.DataLoader(
    #    torch_ds,
    #    batch_size=1,  # TODO is this significantly inhibiting performance
    #    pin_memory=True,
    #    collate_fn=_collate_fn,
    # )
    output_schema = pa.schema(
        [
            # pa.field("row_id", pa.uint64()),
            # TODO these should be some kind of merge id, e.g. "__merge_id"
            # pa.field("id", pa.int64()),
            pa.field("__pq_code", pa.list_(pa.uint8(), list_size=num_sub_vectors)),
        ]
    )

    progress = tqdm(total=num_rows)

    device = kmeans_list[0].device

    # ivf_centroids_array = ivf_centroids_batch["_ivf_centroids"]
    # ivf_centroids = ivf_centroids_array.values.to_numpy().reshape(
    #     len(ivf_centroids_array), ivf_centroids_array.type.list_size
    # )
    # ivf_centroids = torch.from_numpy(ivf_centroids).to(device)

    from lance import dataset as lance_ds

    # Shape: (num_sub_vectors, num_centroids, sub_vector_size)
    centroids_per_sub_vector = torch.stack(
        [kmeans.centroids for kmeans in kmeans_list]
    ).to(device)

    from lance.torch.data import _to_tensor as to_full_tensor

    # def _pq_codes_assignment() -> Iterable[pa.RecordBatch]:
    def pq_codes_assignment(batch):
        with torch.no_grad():
            batch = to_full_tensor(batch)
            # for batch in loader:
            # batch["__residual_vec"]
            vecs = batch.to(device).reshape(
                -1, kmeans_list[0].centroids.shape[1] * len(kmeans_list)
            )

            # ids = batch["id"].reshape(-1)

            # row_ids = batch["_rowid"].reshape(-1)
            # ivf_partitions = batch["partition"].int()
            # residualize
            # old-TODO make this __ivf_part_id instead
            # vecs = vecs - ivf_centroids[ivf_partitions]


            # Higher mem option, probably more performant
            # pq_codes = compute_pq_codes_batch(
            #     vecs,
            #     centroids_per_sub_vector,
            #     num_sub_vectors,
            #     vecs.shape[1] // num_sub_vectors,
            # )

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

            # pq_codes = []
            # for sub_vector, kmeans in enumerate(kmeans_list):
            #    sub_vecs = to_sub_vectors(vecs, num_sub_vectors, sub_vector)
            #    pq_codes.append(kmeans.transform(sub_vecs))
            # pq_codes = torch.stack(pq_codes, dim=1)

            # this is expected to be true, so just assert
            # assert vecs.shape[0] == ids.shape[0]

            # ids = ids.cpu()
            pq_codes = pq_codes.cpu()

            # TODO Ignore any invalid vectors.
            # Commented out to avoid bottleneck for now
            # mask = torch.isfinite(pq_codes).all(dim=1).cpu()
            ## row_ids = row_ids[mask]
            # ids = ids[mask]
            # pq_codes = pq_codes[mask]

            pq_codes = pq_codes.to(torch.uint8)

            pq_values = pa.array(pq_codes.numpy().reshape(-1))
            pq_codes = pa.FixedSizeListArray.from_arrays(pq_values, num_sub_vectors)
            part_batch = pa.RecordBatch.from_arrays(
                [pq_codes],
                # [ids, pq_codes],
                # [row_ids, pq_codes],
                schema=output_schema,
            )

            # if len(part_batch) < len(ids):
            #    logging.warning(
            #        "%s vectors are ignored during pq codes assignment",
            #        len(part_batch) - len(ids),
            #    )

            progress.update(part_batch.num_rows)
            return part_batch
            # yield part_batch

    if "__pq_code" in dataset.schema.names:
        dataset.drop_columns(["__pq_code"])

    @lance.batch_udf(
        output_schema=output_schema,
    )
    def pq_codes_assignment_udf(batch):
        return pq_codes_assignment(batch)

    dataset.add_columns(pq_codes_assignment_udf, read_columns=["__residual_vec"])

    # Save disk space
    if "__residual_vec" in dataset.schema.names:
        dataset.drop_columns(["__residual_vec"])

    # rbr = pa.RecordBatchReader.from_batches(output_schema, _pq_codes_assignment())
    # if dst_dataset_uri is None:
    #    dst_dataset_uri = tempfile.mkdtemp()
    # ds = write_dataset(
    #    rbr,
    #    dst_dataset_uri,
    #    schema=output_schema,
    #    max_rows_per_file=dataset.count_rows(),
    #    data_storage_version="stable",
    # )
    # assert len(ds.get_fragments()) == 1
    # files = ds.get_fragments()[0].data_files()
    # assert len(files) == 1

    progress.close()

    # logging.info("Saved precomputed pq_codes to %s", dst_dataset_uri)

    # pq_codes_dataset = lance_ds(dst_dataset_uri)
    # TODO merge on something that's actually guaranteed to exist
    # dataset.merge(pq_codes_dataset, "_rowid", "row_id") # not supported?
    # dataset.merge(pq_codes_dataset, "id")


def _collate_fn(batch):
    return batch[0]


def compute_partitions(
    dataset: LanceDataset,
    column: str,
    kmeans: Any,  # KMeans
    batch_size: int = 1024 * 10 * 4,
    dst_dataset_uri: Optional[Union[str, Path]] = None,
    allow_cuda_tf32: bool = True,
):
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
    """
    from lance.torch.data import LanceDataset as PytorchLanceDataset

    torch.backends.cuda.matmul.allow_tf32 = allow_cuda_tf32

    num_rows = dataset.count_rows()

    # torch_ds = PytorchLanceDataset(
    #    dataset,
    #    batch_size=batch_size,
    #    with_row_id=True,
    #    columns=[column],
    # )
    # loader = torch.utils.data.DataLoader(
    #    torch_ds,
    #    batch_size=1,  # TODO is this significantly inhibiting performance
    #    pin_memory=True,
    #    collate_fn=_collate_fn,
    # )

    dim = kmeans.centroids.shape[1]

    output_schema = pa.schema(
        [
            # pa.field("row_id", pa.uint64()),
            # pa.field("id", pa.int64()),
            pa.field("__ivf_part_id", pa.uint32()),
            pa.field("__residual_vec", pa.list_(pa.float32(), list_size=dim)),
        ]
    )

    progress = tqdm(total=num_rows)

    from lance.torch.data import _to_tensor as to_full_tensor

    def _partition_assignment(batch):  # -> Iterable[pa.RecordBatch]:
        with torch.no_grad():
            batch = to_full_tensor(batch)
            # for batch in loader:
            # vecs = batch[column].to(kmeans.device).reshape(-1, dim)
            vecs = batch.to(kmeans.device).reshape(-1, dim)

            partitions = kmeans.transform(vecs)
            # ids = batch["id"].reshape(-1)
            # row_ids = batch["_rowid"].reshape(-1)
            # this is expected to be true, so just assert
            # assert vecs.shape[0] == row_ids.shape[0]

            residual_vecs = vecs - kmeans.centroids[partitions]

            # ids = ids.cpu()
            # row_ids = row_ids.cpu()
            partitions = partitions.cpu()
            residual_vecs = residual_vecs.cpu()

            # TODO Ignore any invalid vectors.
            # Also a huge bottleneck, disabled for now
            # mask = (partitions.isfinite()).cpu()
            # ids = ids[mask]
            # row_ids = row_ids[mask]
            # partitions = partitions[mask]
            # residual_vecs = residual_vecs[mask]

            # residual_vecs = residual_vecs.to(vecs.dtype)

            residual_values = pa.array(residual_vecs.numpy().reshape(-1))
            residual_vecs = pa.FixedSizeListArray.from_arrays(residual_values, dim)

            part_batch = pa.RecordBatch.from_arrays(
                [
                    # row_ids.numpy(),
                    # ids.numpy(),
                    partitions.numpy(),
                    residual_vecs,
                ],
                schema=output_schema,
            )
            # if len(part_batch) < len(ids):
            #    logging.warning(
            #        "%s vectors are ignored during partition assignment",
            #        len(part_batch) - len(ids),
            #    )

            progress.update(part_batch.num_rows)
            return part_batch
            # yield part_batch

    @lance.batch_udf(
        output_schema=output_schema,
    )
    def partition_assignment_udf(batch):
        return _partition_assignment(batch)

    # def partition_assignment_iter()-> Iterable[pa.RecordBatch]:
    #    for batch in loader:
    #        yield _partition_assignment(batch)

    if "__residual_vec" in dataset.schema.names:
        dataset.drop_columns(["__residual_vec"])
    if "__ivf_part_id" in dataset.schema.names:
        dataset.drop_columns(["__ivf_part_id"])

    dataset.add_columns(partition_assignment_udf, read_columns=[column])

    # rbr = pa.RecordBatchReader.from_batches(output_schema, partition_assignment_iter())
    # if dst_dataset_uri is None:
    #    dst_dataset_uri = tempfile.mkdtemp()
    # ds = write_dataset(
    #    rbr,
    #    dst_dataset_uri,
    #    schema=output_schema,
    #    max_rows_per_file=dataset.count_rows(),
    #    data_storage_version="stable",
    # )
    # assert len(ds.get_fragments()) == 1
    # files = ds.get_fragments()[0].data_files()
    # assert len(files) == 1

    progress.close()

    dst_dataset_uri = dataset.uri

    logging.info("Saved precomputed partitions to %s", dst_dataset_uri)

    # dataset.merge(ds, "_rowid", "row_id") # not supported I think
    # dataset.merge(ds, "id")

    #return str(dst_dataset_uri)
