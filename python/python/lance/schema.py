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

import json
from typing import Any, Dict

import pyarrow as pa

from .lance import _json_to_schema, _schema_to_json


def schema_to_json(schema: pa.Schema) -> Dict[str, Any]:
    """
    Converts a pyarrow schema to a JSON string.

    Parameters
    ----------
    """
    return json.loads(_schema_to_json(schema))


def json_to_schema(schema_json: Dict[str, Any]) -> pa.Schema:
    """
    Converts a JSON string to a PyArrow schema.

    Parameters
    ----------
    schema_json: Dict[str, Any]
        The JSON payload to convert to a PyArrow Schema.
    """
    return _json_to_schema(json.dumps(schema_json))
