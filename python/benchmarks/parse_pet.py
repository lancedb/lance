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

import numpy as np
import pandas as pd

import pyarrow as pa
from bench_utils import DatasetConverter


class OxfordPetConverter(DatasetConverter):

    def __init__(self, uri_root):
        super(OxfordPetConverter, self).__init__("oxford_pet", uri_root)

    def read_metadata(self) -> pd.DataFrame:
        list_txt = os.path.join(self.uri_root,
                                "annotations/list.txt")
        df = pd.read_csv(list_txt, delimiter=" ", comment="#", header=None)
        df.columns = ["filename", "class", "species", "breed"]

        species_dtype = pd.CategoricalDtype(["Unknown", "Cat", "Dog"])
        df.species = pd.Categorical.from_codes(df.species, dtype=species_dtype)

        breeds = df.filename.str.rsplit("_", 1).str[0].unique()
        assert len(breeds) == 37

        breeds = np.concatenate([["Unknown"], breeds])
        class_dtype = pd.CategoricalDtype(breeds)
        df["class"] = pd.Categorical.from_codes(df["class"], dtype=class_dtype)

        return df

    def default_dataset_path(self, fmt, flavor=None):
        suffix = f"_{flavor}" if flavor else ""
        return os.path.join(self.uri_root,
                            f'{self.name}{suffix}.{fmt}')

    def image_uris(self, table):
        return [os.path.join(self.uri_root, f"images/{x}.jpg")
                for x in table["filename"].to_numpy()]

    def get_schema(self):
        names = ["filename", "class", "species", "breed"]
        types = [pa.string(), pa.string(), pa.string(), pa.int16()]
        return pa.schema([pa.field(name, dtype)
                          for name, dtype in zip(names, types)])
