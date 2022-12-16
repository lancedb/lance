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
"""Dataset conversion"""

import os
from abc import ABC, abstractmethod
from typing import List, Union

import click
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

import lance
from lance.io import download_uris
from lance.types import ImageArray, ImageBinaryType


class DatasetConverter(ABC):
    """Base class for converting raw => pandas => Arrow => Lance"""

    def __init__(self, name, uri_root, images_root: str = None):
        self.name = name
        self.uri_root = uri_root
        if images_root is None:
            images_root = os.path.join(uri_root, "images")
        self.images_root = images_root

    @abstractmethod
    def read_metadata(self, num_rows: int = 0) -> pd.DataFrame:
        """
        Read the metadata / annotations / etc for this dataset

        Parameters
        ----------
        num_rows: int, default 0 which means read all rows
            The number of records to read (used for sampling / testing)

        Return
        ------
        df: pd.DataFrame
            The metadata in this dataset in a dataframe
        """
        pass

    @abstractmethod
    def get_schema(self) -> pa.Schema:
        """Return the Arrow Schema for this Dataset"""
        pass

    @abstractmethod
    def image_uris(self, table) -> List[str]:
        """Return image uris to read the binary column"""
        pass

    def default_dataset_path(self, fmt, flavor=None):
        suffix = f"_{flavor}" if flavor else ""
        return os.path.join(self.uri_root, f"{self.name}{suffix}.{fmt}")

    def write_dataset(self, dataset, output_path, fmt="lance", **kwargs):
        """
        Write the given dataset to either parquet or lance format

        Parameters
        ----------
        dataset: pd.DataFrame | pa.Table | ds.Dataset
            The data to be written
        output_path: str
            The base_dir where to write the data
        fmt: str, 'lance' or 'parquet'
            The data format
        **kwargs: keyword arguments to be passed to `write_dataset`
        """
        if isinstance(dataset, pd.DataFrame):
            dataset = self.to_table(dataset)
        fmt = fmt.lower()
        if fmt == "parquet":
            ds.write_dataset(dataset, output_path, format=fmt, **kwargs)
        elif fmt == "lance":
            lance.write_dataset(dataset, output_path, **kwargs)
        else:
            raise ValueError(f"Unsupported format {fmt}")
        return dataset

    def to_table(self, df: pd.DataFrame, to_image: bool = True) -> pa.Table:
        """Convert each metdata column to pyarrow with lance types"""
        schema = self.get_schema()
        arrays = []
        for name, col in df.items():
            field = schema.field(name)
            arr = self._convert_field(field.name, field.type, col)
            arrays.append(arr)
        table = pa.Table.from_arrays(arrays=arrays, schema=schema).unify_dictionaries()
        if to_image:
            table = self._load_images(table)
        return table

    def _convert_field(self, name, typ, col):
        """pyarrow is unable to convert ExtensionTypes properly in pa.Table.from_pandas"""
        if isinstance(typ, pa.ExtensionType):
            storage = pa.array(col, type=typ.storage_type)
            return pa.ExtensionArray.from_storage(typ, storage)
        elif pa.types.is_list(typ):
            native_arr = pa.array(col)
            if isinstance(native_arr, pa.NullArray):
                return pa.nulls(len(native_arr), typ)
            offsets = native_arr.offsets
            values = native_arr.values.to_numpy(zero_copy_only=False)
            return pa.ListArray.from_arrays(
                offsets, self._convert_field(f"{name}.elements", typ.value_type, values)
            )
        elif pa.types.is_struct(typ):
            native_arr = pa.array(col)
            if isinstance(native_arr, pa.NullArray):
                return pa.nulls(len(native_arr), typ)
            arrays = []
            for subfield in typ:
                sub_arr = native_arr.field(subfield.name)
                converted = self._convert_field(
                    f"{name}.{subfield.name}",
                    subfield.type,
                    sub_arr.to_numpy(zero_copy_only=False),
                )
                arrays.append(converted)
            return pa.StructArray.from_arrays(arrays, fields=typ)
        else:
            return pa.array(col, type=typ)

    def _load_images(self, table: pa.Table, image_col: str = "image"):
        uris = self.image_uris(table)
        images = download_uris(pd.Series(uris))
        image_arr = ImageArray.from_pandas(images)
        embedded = table.append_column(
            pa.field(image_col, ImageBinaryType()), image_arr
        )
        return embedded

    @classmethod
    def create_main(cls):
        FORMATS = click.Choice(["lance", "parquet"])

        @click.command()
        @click.argument("base_uri")
        @click.option(
            "-f",
            "--fmt",
            type=FORMATS,
            default="lance",
            help="Output format (parquet or lance)",
        )
        @click.option(
            "--images-root",
            type=str,
            help="If provided, use this as the uri root for image uri's",
        )
        @click.option("-e", "--embedded", type=bool, default=True, help="Embed images")
        @click.option(
            "-g",
            "--group-size",
            type=int,
            default=1024,
            help="group size",
            show_default=True,
        )
        @click.option(
            "--max-rows-per-file",
            type=int,
            default=0,
            help="max rows per file",
            show_default=True,
        )
        @click.option(
            "-o",
            "--output-path",
            type=str,
            help="Output path. Default is under the base_uri",
        )
        @click.option(
            "--num-rows",
            type=int,
            default=0,
            help="Max rows in the dataset (0 means this is ignored)",
        )
        def main(
            base_uri,
            fmt,
            images_root,
            embedded,
            output_path,
            group_size: int,
            max_rows_per_file: int,
            num_rows: int,
        ):
            converter = cls(base_uri, images_root)
            df = converter.read_metadata(num_rows=num_rows)
            known_formats = ["lance", "parquet"]
            if fmt is not None:
                assert fmt in known_formats
                fmt = [fmt]
            else:
                fmt = known_formats

            for f in fmt:
                if f == "lance":
                    kwargs = {
                        "existing_data_behavior": "overwrite_or_ignore",
                        "max_rows_per_group": group_size,
                        "max_rows_per_file": max_rows_per_file,
                    }
                elif f == "parquet":
                    kwargs = {
                        "max_rows_per_group": group_size,
                    }
                else:
                    raise TypeError(f"Format {f} not supported")
                table = converter.to_table(df, to_image=embedded)
                return converter.write_dataset(table, output_path, f, **kwargs)

        return main
