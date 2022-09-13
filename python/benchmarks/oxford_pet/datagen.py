#!/usr/bin/env python
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

import os
import pathlib
import sys
from urllib.parse import urlparse

sys.path.append("..")

import click
import numpy as np
import pandas as pd
import pyarrow as pa
import xmltodict

from bench_utils import DatasetConverter, download_uris


# Oxford PET has dataset quality issues:
#
# The following exists in the XMLs but are not part of the list.txt index
# Bombay_11,Bombay_189,Bombay_190,Bombay_192,Egyptian_Mau_129,Egyptian_Mau_183,
# Siamese_203,english_cocker_spaniel_162,english_cocker_spaniel_163,
# english_cocker_spaniel_164,english_cocker_spaniel_179,newfoundland_152,
# newfoundland_153,newfoundland_154,newfoundland_155
#
# The following are listed as trainval set but do not have xml annotations:
# Abyssinian_104,Bengal_111,samoyed_10,Bengal_175,Egyptian_Mau_14,
# Egyptian_Mau_156,Egyptian_Mau_186,Ragdoll_199,saint_bernard_15


class OxfordPetConverter(DatasetConverter):
    def __init__(self, uri_root):
        super(OxfordPetConverter, self).__init__("oxford_pet", uri_root)
        self._data_quality_issues = {}

    def read_metadata(self, check_quality=False) -> pd.DataFrame:
        df = self._get_index("list")
        self._to_category(df)
        trainval = self._get_index("trainval")
        i = self._find_split_index(trainval)
        train = pd.Series("train", index=trainval.filename[:i])
        val = pd.Series("val", index=trainval.filename[i:])
        test = self._get_index("test")
        test = pd.Series("test", index=test.filename)
        split = pd.concat([train, val, test])
        split.name = "split"
        split = split.reset_index()
        with_split = df.merge(split, how="left", on="filename")
        xml_files = (
            os.path.join(self.uri_root, "annotations", "xmls/")
            + with_split.filename
            + ".xml"
        )
        ann_df = pd.DataFrame(download_uris(xml_files, func=_get_xml))
        with_xmls = pd.concat([with_split, ann_df.drop(columns=["filename"])], axis=1)

        if check_quality:
            trainval = df[df.split.isin(["train", "val"])]
            self._data_quality_issues["missing_xml"] = trainval[
                trainval.folder.isna()
            ].filename.values.tolist()

            p = pathlib.Path(self.uri_root) / "annotations" / "xmls"
            names = pd.Series([p.name[:-4] for p in p.iterdir()])
            no_index = pd.Index(names.values).difference(df.filename)
            self._data_quality_issues["missing_index"] = no_index

        with_xmls["segmented"] = with_xmls.segmented.apply(
            lambda x: pd.NA if pd.isnull(x) else bool(x)
        ).astype(pd.BooleanDtype())
        return with_xmls

    def _get_index(self, name: str) -> pd.DataFrame:
        list_txt = os.path.join(self.uri_root, f"annotations/{name}.txt")
        df = pd.read_csv(list_txt, delimiter=" ", comment="#", header=None)
        df.columns = ["filename", "class", "species", "breed"]
        return df

    @staticmethod
    def _find_split_index(trainval_df):
        classnames = trainval_df.filename.str.rsplit("_", 1).str[0].str.lower()
        return np.argmax(classnames < classnames.shift(1))

    @staticmethod
    def _to_category(metadata_df: pd.DataFrame):
        species_dtype = pd.CategoricalDtype(["Unknown", "Cat", "Dog"])
        metadata_df["species"] = pd.Categorical.from_codes(
            metadata_df.species, dtype=species_dtype
        )

        breeds = metadata_df.filename.str.rsplit("_", 1).str[0].unique()
        assert len(breeds) == 37

        breeds = np.concatenate([["Unknown"], breeds])
        class_dtype = pd.CategoricalDtype(breeds)
        metadata_df["class"] = pd.Categorical.from_codes(
            metadata_df["class"], dtype=class_dtype
        )
        return metadata_df

    def default_dataset_path(self, fmt, flavor=None):
        suffix = f"_{flavor}" if flavor else ""
        return os.path.join(self.uri_root, f"{self.name}{suffix}.{fmt}")

    def image_uris(self, table):
        return [
            os.path.join(self.uri_root, f"images/{x}.jpg")
            for x in table["filename"].to_numpy()
        ]

    def get_schema(self):
        source_schema = pa.struct(
            [
                pa.field("database", pa.string()),
                pa.field("annotation", pa.string()),
                pa.field("image", pa.string()),
            ]
        )
        size_schema = pa.struct(
            [
                pa.field("width", pa.int32()),
                pa.field("height", pa.int32()),
                pa.field("depth", pa.uint8()),
            ]
        )
        bbox = pa.struct(
            [
                pa.field("xmin", pa.int32()),
                pa.field("ymin", pa.int32()),
                pa.field("xmax", pa.int32()),
                pa.field("ymax", pa.int32()),
            ]
        )
        object_schema = pa.list_(
            pa.struct(
                [
                    pa.field("name", pa.dictionary(pa.uint8(), pa.string())),
                    pa.field("pose", pa.dictionary(pa.uint8(), pa.string())),
                    pa.field("truncated", pa.bool_()),
                    pa.field("occluded", pa.bool_()),
                    pa.field("bndbox", bbox),
                    pa.field("difficult", pa.bool_()),
                ]
            )
        )
        names = [
            "filename",
            "class",
            "species",
            "breed",
            "split",
            "folder",
            "source",
            "size",
            "segmented",
            "object",
        ]
        types = [
            pa.string(),
            pa.dictionary(pa.int8(), pa.string()),
            pa.dictionary(pa.int8(), pa.string()),
            pa.int16(),
            pa.dictionary(pa.int8(), pa.string()),
            pa.string(),
            source_schema,
            size_schema,
            pa.bool_(),
            object_schema,
        ]
        return pa.schema([pa.field(name, dtype) for name, dtype in zip(names, types)])


def _get_xml(uri: str):
    if not urlparse(uri).scheme:
        uri = pathlib.Path(uri)

    fs, key = pa.fs.FileSystem.from_uri(uri)
    try:
        with fs.open_input_file(key) as fh:
            dd = xmltodict.parse(fh.read())["annotation"]
            if not isinstance(dd["object"], list):
                dd["object"] = [dd["object"]]
            sz = dd["size"]
            sz["width"] = int(sz["width"])
            sz["height"] = int(sz["height"])
            sz["depth"] = int(sz["depth"])
            for obj in dd["object"]:
                obj["truncated"] = bool(int(obj["truncated"]))
                obj["occluded"] = bool(int(obj["occluded"]))
                obj["difficult"] = bool(int(obj["difficult"]))
                obj["bndbox"] = {
                    "xmin": int(obj["bndbox"]["xmin"]),
                    "xmax": int(obj["bndbox"]["xmax"]),
                    "ymin": int(obj["bndbox"]["ymin"]),
                    "ymax": int(obj["bndbox"]["ymax"]),
                }
            return dd
    except Exception:
        return {}


@click.command
@click.option(
    "-u", "--base-uri", type=str, required=True, help="Oxford Pet dataset root"
)
@click.option(
    "-f",
    "--fmt",
    type=click.Choice(["parquet", "lance"]),
    help="Output format (parquet or lance)",
)
@click.option("-e", "--embedded", type=bool, default=True, help="store embedded images")
@click.option("-g", "--group-size", type=int, default=1024, help="set max_rows_per_group in arrow")
@click.option("--max-rows-per-file", type=int, default=0, help="set max_rows_per_file in arrow")
@click.option(
    "-o", "--output", type=str, default="oxford_pet.lance", help="Output path"
)
def main(base_uri, fmt, embedded, output, group_size, max_rows_per_file):
    known_formats = ["lance", "parquet"]
    if fmt is not None:
        assert fmt in known_formats
        fmt = [fmt]
    else:
        fmt = known_formats
    converter = OxfordPetConverter(base_uri)
    df = converter.read_metadata()
    for f in fmt:
        if embedded:
            converter.make_embedded_dataset(
                df,
                f,
                output_path=output,
                partitioning=["split"],
                existing_data_behavior="overwrite_or_ignore",
                max_rows_per_group=group_size,
                max_rows_per_file=max_rows_per_file, # Create enough files for parallelism
            )
        else:
            converter.save_df(df, f, output_path=output, partitioning=["split"])


if __name__ == "__main__":
    main()
