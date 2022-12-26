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

from lance.data.experimental.metadata import MLMetadataBuilder, MLMetadata

CLASSIFICATION_JSON = {
    "task_type": "classification",
    "labels": ["species", "breed"],
    "split": "split",
    "ground_truth": {"label": "class"},
    "predictions": [{"name": "resnet", "label": "resnet.label",
                     "score": "resnet.score"},
                    {"name": "vit", "label": "vit.label",
                     "score": "vit.score"}]
}

DETECTION_JSON = {
    "task_type": "detection",
    "split": "split",
    "ground_truth": {"label": "gt.name", "bounding_box": "gt.bbox"},
    "predictions": [{"name": "yolo", "label": "yolo.label",
                     "bounding_box": "yolo.bbox",
                     "score": "yolo.score"}]
}


def test_roundtrip():
    for expected_json in [CLASSIFICATION_JSON, DETECTION_JSON]:
        metadata = MLMetadata.from_json(expected_json)
        assert expected_json == metadata.to_json()
