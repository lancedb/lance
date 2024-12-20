# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

from typing import Any, Dict

import pyarrow as pa

class LanceSchema:
    def to_pyarrow(self) -> pa.Schema: ...
    @staticmethod
    def from_pyarrow(schema: pa.Schema) -> "LanceSchema": ...

def schema_to_json(schema: pa.Schema) -> Dict[str, Any]: ...
def json_to_schema(schema_json: Dict[str, Any]) -> pa.Schema: ...
