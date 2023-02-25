#  Copyright 2023 Lance Developers
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
#  Copyright 2023 Lance Developers
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

"""Tests for predicate pushdown"""
import random
import string

import lance
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest


def create_table(nrows=100):
    intcol = pa.array(range(nrows))
    floatcol = pa.array(np.arange(nrows) * 2/3, type=pa.float32())
    arr = np.arange(nrows) < nrows / 2
    structcol = pa.StructArray.from_arrays([pa.array(arr, type=pa.bool_())], names=["bool"])

    def gen_str(n):
        return ''.join(random.choices("abc"))

    stringcol = pa.array([gen_str(2) for _ in range(nrows)])

    tbl = pa.Table.from_arrays([
        intcol, floatcol, structcol, stringcol
    ], names=["int", "float", "rec", "str"])
    return tbl


@pytest.fixture()
def dataset(tmp_path):
    tbl = create_table()
    yield lance.write_dataset(tbl, tmp_path)


def test_predicates(dataset):
    predicates = [
        pc.field("int") >= 50,
        pc.field("float") < 90.0,
        pc.field("str") != "aa",
        pc.field("str") == "aa",
    ]
    # test simple
    for expr in predicates:
        assert dataset.to_table(filter=expr) == dataset.to_table().filter(expr)

    # test compound
    for expr in predicates:
        for other_expr in predicates:
            compound = expr & other_expr
            assert dataset.to_table(filter=compound) == dataset.to_table().filter(compound)