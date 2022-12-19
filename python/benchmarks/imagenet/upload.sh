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

# Upload Imagenet dataset to S3
#
# ./upload.sh [NUM_RECORDS_PER_SPLIT]

set -e

S3_BASE_DIR="s3://eto-public/datasets/imagenet_1k"
OUTPUT_PATH=imagenet_1k.lance  # this is the root dir of the lance dataset
TARBALL=${OUTPUT_PATH}.tar.gz
LIMIT=${1:-50000}

rm -rf ${OUTPUT_PATH}
rm -rf ${TARBALL}

python ../../lance/data/convert/imagenet.py \
  --limit ${LIMIT} ${OUTPUT_PATH}

pushd ${OUTPUT_PATH}/../
tar -cvf ${TARBALL} ${OUTPUT_PATH}
popd

aws s3 rm --recursive ${S3_BASE_DIR}/${OUTPUT_PATH}
aws s3 cp --recursive $OUTPUT_PATH ${S3_BASE_DIR}/${OUTPUT_PATH}

aws s3 rm ${S3_BASE_DIR}/${TARBALL}
aws s3 cp ${TARBALL} ${S3_BASE_DIR}/${TARBALL}
