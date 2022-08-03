#!/usr/bin/env python3
# Copyright 2022 Lance Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
import multiprocessing as mp
import pandas as pd
import time

import pyarrow.fs

__all__ = ["download_uris", "timeit"]


def get_bytes(uri):
    fs, key = pyarrow.fs.FileSystem.from_uri(uri)
    return fs.open_input_file(key).read()


def download_uris(uris: pd.Series) -> pd.Series:
    pool = mp.Pool(mp.cpu_count() - 1)
    data = pool.map(get_bytes, uris.values)
    return data


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        # first item in the args, ie `args[0]` is `self`
        print(f"Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper
