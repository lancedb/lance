# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import io
from typing import TYPE_CHECKING, Any, Optional, Union

import pyarrow as pa

if TYPE_CHECKING:
    import PIL.Image
    import torch


class HuggingFaceConverter:
    """
    Utility class for from PyArrow RecordBatch to Huggingface internal Type
    """

    def __init__(self, ds_info: dict[str, Any]):
        """Create HuggingFaceConverter from Huggingface dataset info"""
        self.ds_info = ds_info

    def _to_pil_image(self, scalar: pa.StructScalar) -> "PIL.Image.Image":
        import PIL.Image

        row = scalar.as_py()
        if row.get("bytes") is None:
            return PIL.Image.open(row["path"])
        return PIL.Image.open(io.BytesIO(row["bytes"]))

    def to_pytorch(
        self, col: str, array: pa.Array
    ) -> Optional[Union["torch.Tensor", list["PIL.Image.Image"]]]:
        try:
            feature = self.ds_info["info"]["features"][col]
        except KeyError:
            # Not covered in the features
            return None
        if feature["_type"] == "Image":
            return [self._to_pil_image(x) for x in array]
        raise NotImplementedError(
            f"Conversion to {feature['_type']} is not implemented"
        )
