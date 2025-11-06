"""
Virtual environment management for compatibility testing.

Manages creation and execution of test code in isolated virtual environments
with specific Lance versions installed.
"""

import pickle
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
                f"pylance=={self.version}",
                "pytest",
            ],
            check=True,
            capture_output=True,
        )

        self._created = True

    def execute_method(self, obj: Any, method_name: str) -> Any:
        """
        Execute a method on a pickled object in the virtual environment.

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

        # Get path to venv_runner.py
        runner_script = Path(__file__).parent / "venv_runner.py"

        # Pickle the object
        pickled_obj = pickle.dumps(obj)

        # Set PYTHONPATH to include the tests directory so the venv can import
        # test modules. This allows unpickling test classes (they're pickled as
        # forward_compat.*)
        import os

        env = os.environ.copy()
        tests_dir = Path(__file__).parent.parent
        env["PYTHONPATH"] = str(tests_dir)

        # Run the venv_runner.py script
        result = subprocess.run(
            [str(self.python_path), str(runner_script), method_name],
            input=pickled_obj,
            capture_output=True,
            env=env,
        )

        # Parse the result
        if result.returncode == 0:
            response = pickle.loads(result.stdout)
            if response["success"]:
                return response["result"]
            else:
                # This shouldn't happen if returncode is 0, but handle it
                raise RuntimeError(f"Unexpected error: {response}")
        else:
            # Execution failed, unpickle error info
            try:
                error_info = pickle.loads(result.stdout)
                # Re-create the exception with traceback info
                error_msg = (
                    f"Error in venv (Lance {self.version}) calling {method_name}:\n"
                    f"{error_info['exception_type']}: {error_info['exception_msg']}\n"
                    f"\nTraceback from venv:\n{error_info['traceback']}"
                )
                raise RuntimeError(error_msg)
            except (pickle.UnpicklingError, KeyError, EOFError):
                # If we can't unpickle the error, show raw output
                raise RuntimeError(
                    f"Failed to execute {method_name} in venv (Lance {self.version}):\n"
                    f"stdout: {result.stdout.decode('utf-8', errors='replace')}\n"
                    f"stderr: {result.stderr.decode('utf-8', errors='replace')}"
                )

    def cleanup(self):
        """Remove the virtual environment directory."""
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
