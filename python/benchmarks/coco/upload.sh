#!/usr/bin/env bash
#
# Copyright 2022 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Upload COCO dataset to S3

set -e

DATASET_ROOT=$1  # this is the root dir of the raw dataset
OUTPUT_PATH=coco.lance  # this is the root dir of the lance dataset

rm -rf ${OUTPUT_PATH}
rm -rf ${OUTPUT_PATH}.tar.gz

python datagen.py \
  $DATASET_ROOT --output-path $OUTPUT_PATH \
  --fmt lance -e true --max-rows-per-file 10240

pushd ${OUTPUT_PATH}/../
tar -cvf ${OUTPUT_PATH}.lance.tar.gz ${OUTPUT_PATH}
popd

aws s3 rm --recursive s3://eto-public/datasets/coco/coco.lance
aws s3 cp --recursive $OUTPUT_PATH s3://eto-public/datasets/coco/coco.lance

aws s3 rm s3://eto-public/datasets/coco/coco.lance.tar.gz
aws s3 cp ${OUTPUT_PATH}.tar.gz s3://eto-public/datasets/coco/coco.lance.tar.gz
