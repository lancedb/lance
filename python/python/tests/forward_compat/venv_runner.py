"""
Runner script executed inside virtual environments to run compatibility tests.

This script is executed in a subprocess with a specific Lance version installed.
It receives a pickled test object and method name, executes the method, and
returns the result.
"""

import pickle
import sys
import traceback


def main():
    if len(sys.argv) < 2:
        print("Usage: venv_runner.py <method_name>", file=sys.stderr)
        sys.exit(1)

    method_name = sys.argv[1]

    try:
        # Read pickled object from stdin
        obj = pickle.load(sys.stdin.buffer)

        # Call the specified method
        method = getattr(obj, method_name)
        result = method()

        # Write success indicator and optional result
        pickle.dump({"success": True, "result": result}, sys.stdout.buffer)
        sys.exit(0)

    except Exception as e:
        # Capture exception details to send back
        error_info = {
            "success": False,
            "exception_type": type(e).__name__,
            "exception_msg": str(e),
            "traceback": traceback.format_exc(),
        }
        pickle.dump(error_info, sys.stdout.buffer)
        sys.exit(1)


if __name__ == "__main__":
    main()
