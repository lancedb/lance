#!/usr/bin/env bash

#
# Copyright 2023 Lance Developers
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

rm -rf sift1m_*.lance

./datagen.py sift/sift_base.fvecs sift1m_base.lance

cp -r sift1m_base.lance sift1m_ivf512_pq16.lance
./index.py sift1m_ivf512_pq16.lance -i 512 -p 16 -c vector
./metrics.py sift1m_ivf512_pq16.lance lance_ivf512.csv -i 512

cp -r sift1m_base.lance sift1m_ivf1024_pq16.lance
./index.py sift1m_ivf1024_pq16.lance -i 1024 -p 16 -c vector
./metrics.py sift1m_ivf1024_pq16.lance lance_ivf1024.csv -i 1024

cp -r sift1m_base.lance sift1m_ivf2048_pq16.lance
./index.py sift1m_ivf2048_pq16.lance -i 2048 -p 16 -c vector
./metrics.py sift1m_ivf2048_pq16.lance lance_ivf2048.csv -i 2048

python -c "import pandas as pd; pd.concat([pd.read_csv(f'lance_ivf{ivf}.csv') for ivf in [512, 1024, 2048]]).to_csv('lance_sift1m_stats.csv', index=False)"