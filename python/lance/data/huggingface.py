#  Copyright 2022 Lance Developers
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

"""Huggingface adaptor"""

try:
    import datasets
except ImportError as ie:
    raise ImportError("Please install Hugging Face datasets via 'pip install datasets'") from ie

import pyarrow as pa


def features_to_schema(features) -> pa.Schema:
    """Convert Hugging Face features to lance / arrow semantic types."""
    print(features)
    pass
