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
from typing import List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import xmltodict

from lance.io import download_uris
from lance.types import ImageUriType

sys.path.append("..")
from converter import DatasetConverter


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

    def read_metadata(self, num_rows: int = 0, check_quality=False) -> pd.DataFrame:
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
        if num_rows > 0:
            by_split = with_split.groupby("split")
            sizes = by_split.size()
            rows = np.round(sizes / sizes.sum() * num_rows).astype(int).to_dict()
            with_split = pd.concat([f.sample(rows[s]) for s, f in by_split])

        xml_files = (
                os.path.join(self.uri_root, "annotations", "xmls/")
                + with_split.filename
                + ".xml"
        )
        ann_df = pd.DataFrame(download_uris(xml_files, func=_get_xml))
        with_xmls = pd.concat(
            [with_split.reset_index(drop=True), ann_df.drop(columns=["filename"])],
            axis=1,
        )

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

        def _convert(obj_list):
            # change list<struct> to struct<list>
            keys = ["name", "pose", "truncated", "occluded", "bndbox", "difficult"]
            defaults = ["", "", False, False, [], False]
            if isinstance(obj_list, list):
                dd = {}
                for obj in obj_list:
                    for k in keys:
                        dd.setdefault(k, []).append(obj[k])
                return dd
            return dict(zip(keys, [[]] * len(keys)))

        with_xmls["object"] = with_xmls["object"].apply(_convert)
        with_xmls["external_image"] = with_xmls["filename"].apply(
            lambda x: os.path.join(self.uri_root, f"images/{x}.jpg"))
        (with_xmls.reset_index(drop=True)
         .reset_index(names=["_pk"], inplace=True))
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

    def image_uris(self, table) -> List[str]:
        return table["external_image"]

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
        object_schema = pa.struct(
            [
                pa.field("name", pa.list_(pa.string())),
                pa.field("pose", pa.list_(pa.string())),
                pa.field("truncated", pa.list_(pa.bool_())),
                pa.field("occluded", pa.list_(pa.bool_())),
                pa.field("bndbox", pa.list_(pa.list_(pa.float32(), 4))),
                pa.field("difficult", pa.list_(pa.bool_())),
            ]
        )
        names = [
            "_pk",
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
            "external_image",
        ]
        types = [
            pa.int16(),
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
            ImageUriType(),
        ]
        return pa.schema(
            [pa.field(name, dtype) for name, dtype in zip(names, types)],
            metadata={"primary_key": "_pk"},
        )


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
                obj["bndbox"] = [
                    int(obj["bndbox"][pt]) for pt in ["xmin", "ymin", "xmax", "ymax"]
                ]
            return dd
    except Exception:
        return {}


if __name__ == "__main__":
    main = OxfordPetConverter.create_main()
    main()
