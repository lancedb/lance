# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa

from lance.file import LanceFileReader, LanceFileWriter


class IvfModel:
    """
    A class that represents a trained IVF model.
    """

    def __init__(self, centroids: pa.Array, distance_type: str):
        self.centroids = centroids
        """The centroids of the IVF clusters"""
        self.distance_type = distance_type
        """The distance type used to train the IVF model"""

    @property
    def num_partitions(self) -> int:
        """
        The number of partitions / centroids in the IVF model
        """
        return len(self.centroids)

    def save(self, uri: str):
        """
        Save the IVF model to a lance file.

        Parameters
        ----------

        uri: str
            The URI to save the model to.  The URI can be a local file path or a
            cloud storage path.
        """
        with LanceFileWriter(
            uri,
            pa.schema(
                [pa.field("centroids", self.centroids.type)],
                metadata={b"distance_type": self.distance_type.encode()},
            ),
        ) as writer:
            batch = pa.table([self.centroids], names=["centroids"])
            writer.write_batch(batch)

    @classmethod
    def load(cls, uri: str):
        """
        Load an IVF model from a lance file.

        Parameters
        ----------

        uri: str
            The URI to load the model from.  The URI can be a local file path or a
            cloud storage path.
        """
        reader = LanceFileReader(uri)
        num_rows = reader.metadata().num_rows
        metadata = reader.metadata().schema.metadata
        distance_type = metadata[b"distance_type"].decode()
        centroids = (
            reader.read_all(batch_size=num_rows).to_table().column("centroids").chunk(0)
        )
        return cls(centroids, distance_type)
