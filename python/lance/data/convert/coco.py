#  Copyright (c) 2022. Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Convert COCO raw JSON format into PyArrow tables.
"""

import json
from pathlib import Path
from typing import Optional, Union
import pandas as pd

import pyarrow as pa
from lance.io import open_uri

__all__ = ["load_detection"]


def _convert_segmentation(s):
    """Convert segmentation"""
    if isinstance(s, list):
        return {"polygon": s}
    return s


def _instance_anns_to_df(instances_json):
    """Instance / Detection annotations and map to string category names

    Returns
    -------
    ann_df: pd.DataFrame
    """
    df = pd.DataFrame(instances_json['annotations'])
    cat_df = pd.DataFrame(instances_json['categories'])
    df["segmentation"] = df.segmentation.apply(_convert_segmentation)
    df["iscrowd"] = df.iscrowd.astype("bool")
    # Convert coco dataset bounding box [x,y,width,height] to [x0,y0,x1,y1] format.
    df["bbox"] = df.bbox.apply(
        lambda arr: [arr[0], arr[1], arr[0] + arr[2], arr[1] + arr[3]]
    )
    category_df = cat_df.rename({"id": "category_id"}, axis=1)
    return df.merge(category_df, on="category_id")


def load_detection(root: [Path, str],
                   annotations_file: Optional[Union[Path, str]] = None,
                   split: str = "train") -> pa.Table:
    """Load MS Coco Detection dataset into Arrow Table

    Parameters
    ----------
    root : Path or str
        The root directory where images are stored.
    annotations_file : Path or str, optional
        If not provided, treat the dataset as test, the image id will be computed via the filenames.
    split : str
        The split tag
    """
    if annotations_file is None:
        # Test dataset
        pass
    with open_uri(annotations_file) as fobj:
        data = json.load(fobj)
    print(data.keys())


if __name__ == "__main__":
    import click


    @click.command
    @click.argument("uri")
    @click.option("-f", help="output format", type=click.Choice(["lance", "parquet"]), default="lance")
    def main(uri: str):
        load_detection()


    main()
