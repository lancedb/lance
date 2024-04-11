# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import importlib.util

if importlib.util.find_spec("tensorflow") is None:
    raise ImportError(
        "Tensorflow is not installed. Please install tensorflow"
        + " to use lance.tf module.",
    )
