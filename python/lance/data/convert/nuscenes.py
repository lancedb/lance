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

# Nuscenes consists of multiple datasets
# nuimages (images only) and nuscenes (point clouds and images)
import json
import os
import sys
import copy
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa

sys.path.append("..")

SUPPORTED_DATASET_VERSIONS = ["v1.0-mini", "v1.0"]
INSTANCE_DATA_TABLES = ["sample", "sample_data", "object_ann", "surface_ann", "category"]

class NuscenesConverter():
    def __init__(self, uri_root: str, dataset_verson: str):
        # We either support Nuimages (just images and sweeps using camera modality)
        # or Nuscenes (includes radar, camera and lidar).
        # Both datasets come with a mini version that contains no splits
        # and a full size version with splits.
        assert dataset_verson in SUPPORTED_DATASET_VERSIONS, f"Nuscenes converter does not support the dataset version {dataset_verson}."
        self.dataset_version = dataset_verson
        self.has_split = True
        
        self.uri_root = uri_root

        if "mini" in self.dataset_version:
            self.has_split = False 
   
   
    def _get_json_data(self, entity: str):
        """
        Reads the json data for the appropriate entity.
        """
        uri = os.path.join(
            self.uri_root, f"{self.dataset_version}", f"{entity}.json"
        )
        fs, path = pa.fs.FileSystem.from_uri(uri)
        with fs.open_input_file(path) as fobj:
            return json.load(fobj)
    
    def _load_instance_data(self):
        instance_data = {}

        for table in INSTANCE_DATA_TABLES:
            instance_data[table] = self._get_json_data(table)

        return instance_data

    def instances_to_df(self):
        instances_df = []
    
        instance_data = self._load_instance_data()
       
        for sample in instance_data["sample"]:            
            object_annotations = []
            object_labels = []
            
            obj_t = sample["key_camera_token"]
            
            # Collect all the objects in the sample
            for o in instance_data["object_ann"]:
                if o["sample_data_token"] == obj_t:
                    
                    # Collect the category label for the object
                    obj_cat_t = o["category_token"]

                    for c in instance_data["category"]:                        
                        if obj_cat_t == c["token"]:
                            object_annotations.append(o)
                            object_labels.append(c)

            # Get the filename for the sample
            for o in instance_data["sample_data"]:
                if sample["token"] == o["sample_token"]:
                    sample_data = copy.deepcopy(o)
            
            # Create a tuple with all of our joined objects
            instances_df.append({
                "sample": sample,
                "sample_data": sample_data,
                "object_annotations": object_annotations,
                "object_labels": object_labels
            })
        
        return pd.DataFrame(instances_df)
            

                    


