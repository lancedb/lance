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

from abc import abstractmethod, ABC
import os
from typing import Union

import click
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import lance
from lance.io import download_uris
from lance.types import ImageArray, ImageBinaryType


class DatasetConverter(ABC):
    """Base class for converting raw => pandas => Arrow => Lance"""

    def __init__(self, name, uri_root):
        self.name = name
        self.uri_root = uri_root

    @abstractmethod
    def read_metadata(self, num_rows: int = 0) -> pd.DataFrame:
        pass

    def default_dataset_path(self, fmt, flavor=None):
        suffix = f"_{flavor}" if flavor else ""
        return os.path.join(self.uri_root, f"{self.name}{suffix}.{fmt}")

    def save_df(self, df, fmt="lance", output_path=None, **kwargs):
        output_path = output_path or self.default_dataset_path(fmt, "links")
        table = self._convert_metadata_df(df)
        if fmt == "parquet":
            pq.write_table(table, output_path, **kwargs)
        elif fmt == "lance":
            pa.dataset.write_dataset(
                table,
                output_path,
                format=lance.LanceFileFormat(),
                **kwargs,
            )
        return table

    def _convert_metadata_df(self, df: pd.DataFrame) -> pa.Table:
        """Convert each metdata column to pyarrow with lance types"""
        schema = self.get_schema()
        arrays = []
        for name, col in df.items():
            field = schema.field(name)
            arr = self._convert_field(field.name, field.type, col)
            arrays.append(arr)
        table = pa.Table.from_arrays(arrays, schema=schema).unify_dictionaries()
        return table

    def _convert_field(self, name, typ, col):
        if isinstance(typ, pa.ExtensionType):
            storage = pa.array(col, type=typ.storage_type)
            arr = pa.ExtensionArray.from_storage(typ, storage)
        elif pa.types.is_list(typ):
            native_arr = pa.array(col)
            offsets = native_arr.offsets
            values = native_arr.values.to_numpy(zero_copy_only=False)
            return pa.ListArray.from_arrays(
                offsets, self._convert_field(f"{name}.elements", typ.value_type, values)
            )
        elif pa.types.is_struct(typ):
            native_arr = pa.array(col)
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
            arr = pa.array(col, type=typ)
        return arr

    @abstractmethod
    def image_uris(self, table):
        pass

    def make_embedded_dataset(
        self,
        table: Union[pa.Table, pd.DataFrame],
        fmt="lance",
        output_path=None,
        **kwargs,
    ):
        if isinstance(table, pd.DataFrame):
            table = self._convert_metadata_df(table)
        output_path = output_path or self.default_dataset_path(fmt)
        uris = self.image_uris(table)
        images = download_uris(pd.Series(uris))
        image_arr = ImageArray.from_pandas(images)
        embedded = table.append_column(pa.field("image", ImageBinaryType()), image_arr)
        if fmt == "parquet":
            pq.write_table(embedded, output_path, **kwargs)
        elif fmt == "lance":
            pa.dataset.write_dataset(
                embedded, output_path, format=lance.LanceFileFormat(), **kwargs
            )
        return embedded

    @abstractmethod
    def get_schema(self):
        pass

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
            help="Max rows in the dataset (0 means this is ignored)"
        )
        def main(
            base_uri,
            fmt,
            embedded,
            output_path,
            group_size: int,
            max_rows_per_file: int,
            num_rows: int
        ):
            converter = cls(base_uri)
            df = converter.read_metadata(num_rows=num_rows)
            known_formats = ["lance", "parquet"]
            if fmt is not None:
                assert fmt in known_formats
                fmt = [fmt]
            else:
                fmt = known_formats

            for f in fmt:
                if f == 'lance':
                    kwargs = {
                        "existing_data_behavior": "overwrite_or_ignore",
                        "max_rows_per_group": group_size,
                        "max_rows_per_file": max_rows_per_file,
                    }
                elif f == 'parquet':
                    kwargs = {
                        'row_group_size': group_size,
                    }
                if embedded:
                    converter.make_embedded_dataset(df, f, output_path, **kwargs)
                else:
                    return converter.save_df(df, f, output_path, **kwargs)

        return main
