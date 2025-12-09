# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa

from lance.file import LanceFileReader, LanceFileWriter


class PqModel:
    """A class that represents a Product Quantization model.

    Can be saved / loaded to checkpoint progress.
    """

    def __init__(self, num_subvectors: int, codebook: pa.FixedSizeListArray):
        self.num_subvectors = num_subvectors
        """The number of subvectors to divide source vectors into"""
        self.codebook = codebook
        """The centroids of the PQ clusters"""

    @property
    def dimension(self):
        """The dimension of the vectors this model was trained on"""
        return self.codebook.type.list_size

    def save(self, uri: str):
        """
        Save the PQ model to a lance file.

        Parameters
        ----------

        uri: str
            The URI to save the model to.  The URI can be a local file path or a
            cloud storage path.
        """
        with LanceFileWriter(
            uri,
            pa.schema(
                [pa.field("codebook", self.codebook.type)],
                metadata={b"num_subvectors": str(self.num_subvectors).encode()},
            ),
        ) as writer:
            batch = pa.table([self.codebook], names=["codebook"])
            writer.write_batch(batch)

    @classmethod
    def load(cls, uri: str):
        """
        Load a PQ model from a lance file.

        Parameters
        ----------

        uri: str
            The URI to load the model from.  The URI can be a local file path or a
            cloud storage path.
        """
        reader = LanceFileReader(uri)
        num_rows = reader.metadata().num_rows
        metadata = reader.metadata().schema.metadata
        num_subvectors = int(metadata[b"num_subvectors"].decode())
        codebook = (
            reader.read_all(batch_size=num_rows).to_table().column("codebook").chunk(0)
        )
        return cls(num_subvectors, codebook)
