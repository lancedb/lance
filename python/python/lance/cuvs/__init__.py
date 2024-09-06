# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import importlib.util

if importlib.util.find_spec("cuvs") is None:
    raise ImportError(
        "cuvs is not installed. Please install cuvs"
        + " to use lance.cuvs module (optional dependency name: "
        + '"cuvs-py39"/"cuvs-py310"/"cuvs-py311")',
    )

if importlib.util.find_spec("pylibraft") is None:
    raise ImportError(
        "pylibraft is not installed. Please install pylibraft"
        + " to use lance.cuvs module (optional dependency name: "
        + '"cuvs-py39"/"cuvs-py310"/"cuvs-py311")',
    )
