# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

import pyarrow as pa

class LanceSchema:
    def to_pyarrow(self) -> pa.Schema: ...
    def from_pyarrow(schema: pa.Schema) -> "LanceSchema": ...
