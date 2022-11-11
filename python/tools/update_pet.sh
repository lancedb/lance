#!/usr/bin/env bash
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
#

set -e

PUBLIC_URI_ROOT="https://eto-public.s3.us-west-2.amazonaws.com/datasets/oxford_pet/"

DATASET_ROOT=$1  # this is the root dir of the raw dataset
OUTPUT_PATH=$2  # this is the root dir of the lance/parquet dataset


python lance/data/convert/oxford_pet.py \
  $DATASET_ROOT --output-path $OUTPUT_PATH \
  --fmt lance --images-root $PUBLIC_URI_ROOT

aws s3 rm --recursive s3://eto-public/datasets/oxford_pet/oxford_pet.lance
aws s3 cp --recursive $OUTPUT_PATH s3://eto-public/datasets/oxford_pet/oxford_pet.lance