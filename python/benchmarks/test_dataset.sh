#!/usr/bin/env bash
# Copyright (c) 2022. Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -x

DATASET=$1
NROWS=$2

pushd ./"${DATASET}" || exit 1

./datagen.py s3://eto-public/datasets/"${DATASET}" --num-rows "${NROWS}" -o /tmp/"${DATASET}".lance
python3 -c "import lance; assert len(lance.dataset('/tmp/${DATASET}.lance').to_table()) == ${NROWS}"
./analytics.py /tmp -f lance