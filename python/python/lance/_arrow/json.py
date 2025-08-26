# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright The Lance Authors

"""JSON support for Lance using JSONB binary format."""

import pyarrow as pa

try:
    import pandas as pd
except ImportError:
    pd = None


class JsonType(pa.ExtensionType):
    """PyArrow extension type for JSON data stored as JSONB."""

    def __init__(self):
        pa.ExtensionType.__init__(self, pa.large_binary(), "lance.json")

    def __arrow_ext_serialize__(self):
        return b""

    @classmethod
    def __arrow_ext_deserialize__(cls, storage_type, serialized):
        return JsonType()

    def __arrow_ext_class__(self):
        return JsonArray


class JsonArray(pa.ExtensionArray):
    """JSON PyArrow Array using JSONB binary format.

    This wraps Rust's JsonArray which handles JSONB encoding/decoding internally.
    """

    @classmethod
    def from_json_strings(cls, values):
        """Create JsonArray from JSON strings.

        Parameters
        ----------
        values : list or pa.Array
            List of JSON strings or PyArrow string array

        Returns
        -------
        JsonArray
            Array with JSONB-encoded data
        """
        # Import the Rust function
        from ..lance import json_array

        # Convert to PyArrow array if needed
        if not isinstance(values, pa.Array):
            values = pa.array(values, type=pa.string())

        # Call Rust function to create JSONB-encoded array
        # The Rust json_array function will use JsonArray::try_from internally
        return json_array(values)

    @classmethod
    def from_pandas(cls, values):
        """Create from pandas Series of JSON strings."""
        # Convert pandas Series to list
        if hasattr(values, "tolist"):
            values = values.tolist()

        # Ensure all values are JSON strings
        import json

        json_strings = []
        for val in values:
            if val is None or (pd and isinstance(val, float) and pd.isna(val)):
                json_strings.append(None)
            else:
                if not isinstance(val, str):
                    val = json.dumps(val)
                json_strings.append(val)

        return cls.from_json_strings(json_strings)


# Register the extension type with PyArrow
pa.register_extension_type(JsonType())


def json_field(name: str, nullable: bool = True) -> pa.Field:
    """Create a PyArrow field with JSON extension type.

    Parameters
    ----------
    name : str
        Field name
    nullable : bool, optional
        Whether the field can contain null values, default True

    Returns
    -------
    pa.Field
        Field with JSON extension type
    """
    return pa.field(name, JsonType(), nullable=nullable)
