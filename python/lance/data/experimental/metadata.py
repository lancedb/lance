#  Copyright 2022 Lance Developers
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations


class MLMetadata:
    """
    Semantic metadata for the dataset indicating things like:
    - task type (classification, detection, etc)
    - which column is ground truth
    - which columns are model predictions
    - which columns are the labels

    And convertible to/from JSON. Validation is done separately by
    a validator that has a reference to the dataset (so can verify
    column names etc)

    Example
    -------
    {
      "type": "classification",
      "ground_truth": {"label": "gt.label"}
      "predictions": [{"name": "resnet", "label": "resnet.label"}]
      "split": "split"
      "labels": ["timeofday", "city"]
    }

    {
      "type": "detection",
      "ground_truth": {"label": "gt.label", "bounding_box": "gt.bbox"},
      "predictions": [{"name": "yolo", "label": "yolo.label",
                       "bounding_box": "yolo.bbox", "score": "yolo.score" }]
      "split": "split"
      "labels": ["timeofday", "city"]
    }
    """
    def __init__(self, task_type: str,
                 ground_truth: dict = None,
                 predictions: list[dict] = None,
                 split: str = None,
                 labels: list[str] = None):
        self._task_type = task_type
        self._ground_truth = ground_truth
        self._predictions = predictions
        self._split = split
        self._labels = labels

    @property
    def task_type(self) -> str:
        """Type of the task (e.g., classification, detection, etc)"""
        return self._task_type

    @property
    def split(self) -> str:
        """Name of the split column"""
        return self._split

    @property
    def labels(self) -> list[str]:
        """Name of columns used as asset-level labels"""
        return None if not self._labels else self._labels.copy()

    @property
    def ground_truth(self) -> dict:
        """Schema for the default ground-truth labels"""
        return None if not self._ground_truth else self._ground_truth.copy()

    @property
    def predictions(self) -> list[dict]:
        """Schema for model predictions"""
        return None if not self._predictions else [m.copy() for m in self._predictions]

    def to_json(self) -> dict:
        dd = { "task_type": self.task_type }
        if self.split:
            dd["split"] = self.split
        if self.labels:
            dd["labels"] = self.labels
        if self.ground_truth:
            dd["ground_truth"] = self.ground_truth
        if self.predictions:
            dd["predictions"] = self.predictions
        return dd

    @classmethod
    def from_json(cls, json_data: dict) -> MLMetadata:
        kwargs = cls.parse_params(json_data)
        return MLMetadata(**kwargs)

    @classmethod
    def parse_params(cls, json_data):
        return {k.strip().lower(): v for k, v in json_data.items()}


class MLMetadataBuilder:

    TASK_TYPES = ["classification", "detection"]

    def __init__(self, dataset: "FileSystemDataset"):
        self.dataset = dataset
        self._task_type = None
        self._labels = None
        self._split = None
        self._predictions = None
        self._ground_truth = None

    def task_type(self, task_type: str) -> MLMetadataBuilder:
        self._task_type = task_type.lower().strip()
        return self

    def labels(self, labels: list[str]) -> MLMetadataBuilder:
        self._labels = labels
        return self

    def split(self, split: str) -> MLMetadataBuilder:
        self._split = split
        return self

    def predictions(self, predictions: list[str]) -> MLMetadataBuilder:
        self._predictions = predictions
        return self

    def ground_truth(self, gt: dict) -> MLMetadataBuilder:
        self._ground_truth = gt
        return self

    def to_metadata(self) -> MLMetadata:
        self.validate()
        return MLMetadata(task_type=self._task_type,
                          labels=self._labels,
                          split=self._split,
                          ground_truth=self._ground_truth,
                          predictions=self._predictions)

    def validate(self):
        typ = self._task_type
        if typ not in self.TASK_TYPES:
            err = f"Only {self.TASK_TYPES} supported, but got {typ}."
            raise ValueError(err)

        self._check_colnames()
        if self._predictions and not self._ground_truth:
            raise TypeError("Cannot have predictions without ground truth column")

        if self._ground_truth:
            if ((typ == "detection") and
                    ("bounding_box" not in self._ground_truth)):
                raise TypeError("Detection ground_truth need bounding_box")

        if self._predictions:
            for p in self._predictions:
                if (typ == "detection") and ("bounding_box" not in p):
                    raise TypeError(f"Detection {p} need bounding_box")

    def _check_colnames(self):
        cols = []
        if self._labels:
            cols.extend(self._labels)
        if self._split:
            cols.append(self._split)
        if self._predictions:
            cols.extend(self._get_flattened(self._predictions))
        if self._ground_truth:
            cols.extend(self._get_flattened([self._ground_truth]))
        names = set(self.dataset.schema.names)
        for c in cols:
            if c not in names:
                raise ValueError(f"{c} not found in dataset")

    @staticmethod
    def _get_flattened(lst):
        colnames = [[p.get("label"), p.get("bounding_box"), p.get("score")]
                    for p in lst]
        colnames = [fname for sublist in colnames
                    for fname in sublist if fname]
        return colnames
