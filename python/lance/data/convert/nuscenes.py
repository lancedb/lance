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

import copy
import json
import os
import sys
import urllib.parse
from collections import defaultdict
from typing import Iterable, List

import numpy as np
import pandas as pd
import pyarrow as pa

import lance
from lance.data.convert.base import DatasetConverter
from lance.types import Box2dType

sys.path.append("..")

SUPPORTED_DATASET_VERSIONS = ["v1.0-mini", "v1.0", "v1.0-test", "v1.0-train", "v1.0-val"]
INSTANCE_DATA_TABLES = [
    "sample",
    "sample_data",
    "object_ann",
    "surface_ann",
    "category",
    "ego_pose",
    "calibrated_sensor",
    "sensor",
    "log"
]
FOREIGN_KEYS = {
    "sample": ["log_token", "key_camera_token"],
    "sample_data": ["sample_token", "ego_pose_token", "calibrated_sensor_token"],
    "object_ann": ["sample_data_token", "category_token", "attribute_tokens"],
    "surface_ann": ["sample_data_token", "category_token"],
    "category": [""],
    "ego_pose": [""],
    "calibrated_sensor": ["sensor_token"],
    "sensor": [""],
    "log": [""]
}

class NuscenesConverter(DatasetConverter):
    def __init__(self, uri_root: str, images_root: str, dataset_verson: str):
        """We either support Nuimages (just images and sweeps using camera modality)
        or Nuscenes (includes radar, camera and lidar).
        Both datasets come with a mini version that contains no splits
        and a full size version with splits.
       
        """
        super(NuscenesConverter, self).__init__(
            "nuscenes", uri_root, images_root)

        assert dataset_verson in SUPPORTED_DATASET_VERSIONS, f"Nuscenes converter does not support the dataset version {dataset_verson}."
        self.dataset_version = dataset_verson
        self.has_split = True
        self.instance_data = self._load_instance_data()
        self.split = ""

        if "mini" in self.dataset_version:
            self.has_split = False
        elif "test" in self.dataset_version:
            self.split = "test"
        elif "train" in self.dataset_version:
            self.split = "train"
        elif "val" in self.dataset_version:
            self.split = "val"
        
    def _get_json_data(self, entity: str):
        """Reads the json data for the appropriate entity."""
        uri = os.path.join(
            self.uri_root, f"{self.dataset_version}", f"{entity}.json"
        )
        fs, path = pa.fs.FileSystem.from_uri(uri)
        with fs.open_input_file(path) as fobj:
            return json.load(fobj)
    
    def _load_instance_data(self):
        """ Loads the json data."""
        instance_data = {}

        for table in INSTANCE_DATA_TABLES:
            instance_data[table] = self._get_json_data(table)

        return instance_data

    def _clone_and_rename(self, object, table_name, prefix_table_name = True, strip_pk = True):
        """Renames the object, as when we merge to flatten
        we lose semantic information, thus we prefix the column
        name with the table name we are flattening from.

        """
        object_joined = {}
        for key in object:
            fks = FOREIGN_KEYS[table_name]
            if ((key == "token" and strip_pk) or key in fks):
                continue
            if (prefix_table_name):
                object_joined[f"{table_name}_{key}"] = object[key]
            else:
                object_joined[key] = object[key]
        return object_joined

    def _merge_dict(self, obj1, obj2, table_name):
        """ Merges two dictionaries to flatten the object."""
        rename_and_cloned_obj2 = self._clone_and_rename(obj2, table_name)
        return {**obj1, **rename_and_cloned_obj2}

    def _find_foreign_object(self, token, foreign_key, table_name):
        """Find and returns a foreign object with a given foreign key
        and token value.

        """
        for object in self.instance_data[table_name]:
            if token == object[foreign_key]:
                return object
        return None

    def _find_annotations(self, token, foreign_key, table_name):
        """Collects the annotation for the given sample."""
        objects = []
        for object in self.instance_data[table_name]:
            if token == object[foreign_key]:
                category = self._find_foreign_object(object["category_token"], "token", "category")
                object_joined = self._merge_dict(object, category, "category")
                objects.append(object_joined)
        return objects

    def read_metadata(self, num_rows: int = 0):
        """Converts the dataset to a dataframe that is flattened.
        To do so, we need to get all the samples and
        join various entities to the sample.

        """
        instances_df = []
        current_row = 0
           
        for sample in self.instance_data["sample"]:
            if num_rows > current_row or num_rows == 0:
                current_row += 1

                # Find all related entities to the sample
                log_data = self._find_foreign_object(sample["log_token"], "token", "log")
                sample_data = self._find_foreign_object(sample["token"], "sample_token", "sample_data")
                ego_pose = self._find_foreign_object(sample_data["ego_pose_token"], "token", "ego_pose")
                calibrated_sensor = self._find_foreign_object(sample_data["calibrated_sensor_token"], "token", "calibrated_sensor")
                sensor = self._find_foreign_object(calibrated_sensor["sensor_token"], "token", "sensor")

                # Create the flatten sample
                sample_joined = self._clone_and_rename(sample, "sample", False, False)
                sample_joined["split"] = self.split

                sample_joined = self._merge_dict(sample_joined, log_data, "log")
                sample_joined = self._merge_dict(sample_joined, sample_data, "sample_data")
                sample_joined = self._merge_dict(sample_joined, ego_pose, "ego_pose")
                sample_joined = self._merge_dict(sample_joined, calibrated_sensor, "calibrated_sensor")
                sample_joined = self._merge_dict(sample_joined, sensor, "sensor")

                # Find the annotations for the sample
                object_anns = self._find_annotations(sample_data["token"], "sample_data_token", "object_ann")
                surface_ann = self._find_annotations(sample_data["token"] , "sample_data_token", "surface_ann")
                sample_joined["object_ann"] = object_anns
                sample_joined["surface_ann"] = surface_ann

                # Encode a URL for external ref images
                sample_joined["image_url"] = urllib.parse.quote(sample_joined["sample_data_filename"])
                
                instances_df.append(sample_joined)
        
        return pd.DataFrame(instances_df)
    
    def get_schema(self) -> pa.Schema:
        """Returns the Arrow schema.
        This is flatten/denormalized for better ergonomics for
        ML - i.e. filtering/slicing the dataset and reducing joins.
        Column names are renamed accordingly to prevent collisions.

        """
        # Mask
        mask_schema = pa.struct([
            pa.field("size", pa.list_(pa.int32())),
            pa.field("counts", pa.string())
        ])

        # Surface Ann
        surface_ann_schema = pa.struct([
            pa.field("mask", mask_schema), # TODO: change to RLE once type is created 

            # Category
            pa.field("category_name", pa.string()),
            pa.field("category_description", pa.string())
        ])

        # Object Ann
        object_ann_schema = pa.struct([
            pa.field("bbox", Box2dType()),
            pa.field("mask", mask_schema), # TODO: change to RLE once type is created

            # Category
            pa.field("category_name", pa.string()),
            pa.field("category_description", pa.string())
        ])

        return pa.schema([
            pa.field("split", pa.string()),

            # Sample
            pa.field("token", pa.string()),
            pa.field("timestamp", pa.timestamp('us')),
            
            # Log
            pa.field("log_logfile", pa.string()),
            pa.field("log_vehicle", pa.string()),
            pa.field("log_date_captured", pa.string()),
            pa.field("log_location", pa.string()),
            
            # Calibrated Sensor
            pa.field("calibrated_sensor_translation", pa.list_(pa.float32())),
            pa.field("calibrated_sensor_rotation", pa.list_(pa.float32())),
            pa.field("calibrated_sensor_camera_intrinsic", pa.list_(pa.list_(pa.float32()))),
            pa.field("calibrated_sensor_camera_distortion", pa.list_(pa.float32())), # can be 5 or 6 length
            
            # Sensor
            pa.field("sensor_channel", pa.string()),
            pa.field("sensor_modality", pa.string()),

            # Sample Data
            pa.field("sample_data_filename", pa.string()),
            pa.field("sample_data_fileformat", pa.string()),
            pa.field("sample_data_width", pa.int32()),
            pa.field("sample_data_height", pa.int32()),
            pa.field("sample_data_timestamp", pa.timestamp('us')),
            pa.field("sample_data_is_key_frame", pa.bool_()),
            pa.field("sample_data_next", pa.string()),
            pa.field("sample_data_prev", pa.string()),

            # Ego Pose
            pa.field("ego_pose_translation", pa.list_(pa.float32())),
            pa.field("ego_pose_rotation", pa.list_(pa.float32())),
            pa.field("ego_pose_timestamp", pa.timestamp('us')),
            pa.field("ego_pose_rotation_rate", pa.list_(pa.float32())),
            pa.field("ego_pose_acceleration", pa.list_(pa.float32())),
            pa.field("ego_pose_speed", pa.float32()),

            # Annotations
            pa.field("surface_ann", pa.list_(surface_ann_schema)),
            pa.field("object_ann", pa.list_(object_ann_schema)),

            # Image URL
            pa.field("image_url", pa.string())
        ])
    
    def image_uris(self, table) -> List[str]:
        """Return image uris to read the binary column"""
        uris = [os.path.join(self.images_root, image_uri)
            for image_uri in table["sample_data_filename"].to_numpy()]
        return uris