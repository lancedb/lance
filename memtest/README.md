# lance-memtest

Memory allocation testing utilities for Python test suites. This package provides tools to track memory allocations made by the Python interpreter and any Python libraries during test execution.

## Usage

Install with:

```shell
make build-release
```

To activate the memory tracking, you need to set the `LD_PRELOAD` environment variable:

```shell
export LD_PRELOAD=$(lance-memtest)
```

On macOS, use `DYLD_INSERT_LIBRARIES` instead:

```shell
export DYLD_INSERT_LIBRARIES=$(lance-memtest)
```

Then you can write Python code that tracks memory allocations:

```python
import memtest

def test_memory():
    with memtest.track() as get_stats:
        # Your code that allocates memory
        data = [0] * 1000000

        stats = get_stats()
        assert stats['peak_bytes'] < 10**7  # Assert peak memory usage
```

## How this works

The library uses dynamic linking to intercept memory allocation calls (like `malloc`, `free`, etc.) made by the Python interpreter and its extensions. It keeps track of the total number of allocations, deallocations, and the peak memory usage during the execution of your code.
