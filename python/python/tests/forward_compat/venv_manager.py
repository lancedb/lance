"""
Virtual environment management for compatibility testing.

Manages creation and execution of test code in isolated virtual environments
with specific Lance versions installed.
"""

import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


class VenvExecutor:
    """Manages a virtual environment with a specific Lance version."""

    def __init__(self, version: str, venv_path: Path):
        """
        Initialize a VenvExecutor.

        Parameters
        ----------
        version : str
            Lance version to install (e.g., "0.30.0")
        venv_path : Path
            Directory where virtual environment will be created
        """
        self.version = version
        self.venv_path = Path(venv_path)
        self.python_path: Optional[Path] = None
        self._created = False
        self._subprocess: Optional[subprocess.Popen] = None

    def create(self):
        """Create the virtual environment and install the specified Lance version."""
        if self._created:
            return

        # Create virtual environment
        subprocess.run(
            [sys.executable, "-m", "venv", str(self.venv_path)],
            check=True,
            capture_output=True,
        )

        # Determine python path in venv
        if sys.platform == "win32":
            self.python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            self.python_path = self.venv_path / "bin" / "python"

        # Upgrade pip
        subprocess.run(
            [str(self.python_path), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
        )

        # Install specific pylance version and pytest (needed for test modules)
        subprocess.run(
            [
                str(self.python_path),
                "-m",
                "pip",
                "install",
                "--pre",
                "--extra-index-url",
                "https://pypi.fury.io/lancedb/",
                f"pylance=={self.version}",
                "pytest",
            ],
            check=True,
            capture_output=True,
        )

        self._created = True

    def _ensure_subprocess(self):
        """Ensure the persistent subprocess is running."""
        if self._subprocess is not None and self._subprocess.poll() is None:
            # Subprocess is already running
            return

        # Start persistent subprocess
        runner_script = Path(__file__).parent / "venv_runner.py"

        # Set PYTHONPATH to include the tests directory
        env = os.environ.copy()
        tests_dir = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(tests_dir)

        self._subprocess = subprocess.Popen(
            [str(self.python_path), "-u", str(runner_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )

    def _send_message(self, obj: Any):
        """Send a length-prefixed pickled message to subprocess."""
        data = pickle.dumps(obj)
        length = struct.pack(">I", len(data))
        self._subprocess.stdin.write(length)
        self._subprocess.stdin.write(data)
        self._subprocess.stdin.flush()

    def _receive_message(self) -> Any:
        """Receive a length-prefixed pickled message from subprocess."""
        # Read 4-byte length header
        length_bytes = self._subprocess.stdout.read(4)
        if len(length_bytes) < 4:
            raise RuntimeError("Failed to read message length from subprocess")

        length = struct.unpack(">I", length_bytes)[0]

        # Read message data
        data = self._subprocess.stdout.read(length)
        if len(data) < length:
            raise RuntimeError(
                f"Incomplete message: expected {length} bytes, got {len(data)}"
            )

        return pickle.loads(data)

    def execute_method(self, obj: Any, method_name: str) -> Any:
        """
        Execute a method on a pickled object in the virtual environment.

        Uses a persistent subprocess to avoid repeatedly importing Lance and
        its dependencies.

        Parameters
        ----------
        obj : Any
            Object to pickle and send to venv. Must be picklable.
        method_name : str
            Name of the method to call on the object

        Returns
        -------
        Any
            Return value from the method call

        Raises
        ------
        Exception
            Re-raises any exception that occurred in the venv
        """
        if not self._created:
            raise RuntimeError("Virtual environment not created. Call create() first.")

        # Ensure subprocess is running
        self._ensure_subprocess()

        try:
            # Send request: (obj, method_name)
            self._send_message((obj, method_name))

            # Receive response
            response = self._receive_message()

            if response["success"]:
                return response["result"]
            else:
                # Error occurred in subprocess
                error_msg = (
                    f"Error in venv (Lance {self.version}) calling {method_name}:\n"
                    f"{response['exception_type']}: {response['exception_msg']}\n"
                    f"\nTraceback from venv:\n{response['traceback']}"
                )
                raise RuntimeError(error_msg)

        except (BrokenPipeError, EOFError, struct.error) as e:
            # Subprocess died or communication failed
            stderr_output = ""
            if self._subprocess and self._subprocess.stderr:
                stderr_output = self._subprocess.stderr.read().decode(
                    "utf-8", errors="replace"
                )

            raise RuntimeError(
                f"Communication with venv subprocess failed (Lance {self.version}):\n"
                f"Error: {e}\n"
                f"stderr: {stderr_output}"
            )

    def cleanup(self):
        """Remove the virtual environment directory and terminate subprocess."""
        # Terminate the persistent subprocess
        if self._subprocess is not None:
            try:
                self._subprocess.stdin.close()
                self._subprocess.terminate()
                self._subprocess.wait(timeout=5)
            except Exception:
                # Force kill if graceful termination fails
                self._subprocess.kill()
            finally:
                self._subprocess = None

        # Remove venv directory
        if self.venv_path.exists():
            import shutil

            shutil.rmtree(self.venv_path)
        self._created = False


class VenvFactory:
    """Factory for creating and managing VenvExecutor instances."""

    def __init__(self, base_path: Path):
        """
        Initialize the factory.

        Parameters
        ----------
        base_path : Path
            Base directory for creating virtual environments
        """
        self.base_path = Path(base_path)
        self.venvs: dict[str, VenvExecutor] = {}

    def get_venv(self, version: str) -> VenvExecutor:
        """
        Get or create a VenvExecutor for the specified version.

        Parameters
        ----------
        version : str
            Lance version

        Returns
        -------
        VenvExecutor
            Executor for the specified version
        """
        if version not in self.venvs:
            venv_path = self.base_path / f"venv_{version}"
            executor = VenvExecutor(version, venv_path)
            executor.create()
            self.venvs[version] = executor
        return self.venvs[version]

    def cleanup_all(self):
        """Clean up all created virtual environments."""
        for venv in self.venvs.values():
            venv.cleanup()
        self.venvs.clear()
