"""
Runner script executed inside virtual environments to run compatibility tests.

This script runs as a persistent subprocess that accepts multiple method calls
without restarting. This avoids the overhead of repeatedly importing Lance and
its dependencies.

Protocol:
- Reads 4 bytes (message length as big-endian int)
- Reads that many bytes (pickled tuple of (obj, method_name))
- Executes method on object
- Writes 4 bytes (response length)
- Writes pickled response dict
"""

import pickle
import struct
import sys
import traceback


def read_message(stream):
    """Read a length-prefixed pickled message from stream."""
    # Read 4-byte length header
    length_bytes = stream.buffer.read(4)
    if len(length_bytes) < 4:
        return None  # EOF

    length = struct.unpack(">I", length_bytes)[0]

    # Read message data
    data = stream.buffer.read(length)
    if len(data) < length:
        raise RuntimeError(
            f"Incomplete message: expected {length} bytes, got {len(data)}"
        )

    return pickle.loads(data)


def write_message(stream, obj):
    """Write a length-prefixed pickled message to stream."""
    data = pickle.dumps(obj)
    length = struct.pack(">I", len(data))
    stream.buffer.write(length)
    stream.buffer.write(data)
    stream.buffer.flush()


def main():
    """Main loop that processes method calls until EOF."""
    while True:
        try:
            # Read request (obj, method_name)
            request = read_message(sys.stdin)
            if request is None:
                # EOF - parent closed connection
                break

            obj, method_name = request

            # Execute method
            method = getattr(obj, method_name)
            result = method()

            # Send success response
            response = {"success": True, "result": result}
            write_message(sys.stdout, response)

        except Exception as e:
            # Send error response
            error_info = {
                "success": False,
                "exception_type": type(e).__name__,
                "exception_msg": str(e),
                "traceback": traceback.format_exc(),
            }
            write_message(sys.stdout, error_info)


if __name__ == "__main__":
    main()
