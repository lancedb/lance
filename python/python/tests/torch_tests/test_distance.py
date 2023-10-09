#  Copyright (c) 2023. Lance Developers
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

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_cosine_distance():
    from lance.torch.distance import cosine_distance

    x = np.random.randn(20, 256).astype(np.float32)
    y = np.random.rand(100, 256).astype(np.float32)

    dist = cosine_distance(torch.from_numpy(x), torch.from_numpy(y))
    assert dist.shape == (20, 100)

    # Brute-force / the simplest proof.
    expect = []
    for x_row in x:
        for y_row in y:
            expect.append(
                1 - np.dot(x_row, y_row) / np.linalg.norm(x_row) / np.linalg.norm(y_row)
            )
    expect_arr = np.array(expect).astype(np.float32).reshape(20, 100)
    assert np.allclose(dist.cpu(), expect_arr)


def test_l2_distance():
    from lance.torch.distance import l2_distance

    x = np.random.randn(20, 256).astype(np.float32)
    y = np.random.rand(100, 256).astype(np.float32)

    dist = l2_distance(torch.from_numpy(x), torch.from_numpy(y))
    assert dist.shape == (20, 100)
    expect = []
    for x_row in x:
        for y_row in y:
            expect.append(np.linalg.norm(x_row - y_row))
    expect_arr = np.array(expect).astype(np.float32).reshape(20, 100)
    assert np.allclose(dist.cpu(), expect_arr)
