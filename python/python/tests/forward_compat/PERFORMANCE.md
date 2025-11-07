# Compatibility Test Performance Analysis

## Persistent Virtual Environments (Default)

By default, virtual environments are **persistent** and stored in `~/.cache/lance-compat-venvs/`.

**First run (creates venv):** ~13-16s per version
**Subsequent runs (reuses venv):** ~2-6s per version

This makes interactive development much faster - you only pay the setup cost once!

## Timing Breakdown (per version, first test with venv creation)

```
Total: ~16-17s
├── Virtual environment setup: ~7-8s (47%)
│   ├── venv creation: 2.2s
│   └── package install (pylance + pytest): 4.9s
├── First Lance import in subprocess: ~5.0s (29%)
├── Test execution: ~0.04s (0.2%)
└── Overhead (pytest, data creation, etc.): ~4s (24%)
```

## Timing Breakdown (subsequent runs with persistent venv)

```
Total: ~2-6s
├── Venv validation: ~0.1s
├── Lance import (if new subprocess): ~2-5s
├── Test execution: ~0.04s
└── Overhead: ~1-2s
```

## What's Working Well

1. **Persistent subprocess**: Subsequent method calls are 500x faster (5s → 0.01s)
2. **Venv reuse**: Virtual environments are cached across tests in same session
3. **Pip cache**: Leveraging ~11GB pip cache for faster installs

## Current Optimizations

- ✅ **Persistent virtual environments** (default, 5x speedup for subsequent runs)
- ✅ Removed pip upgrade step (saves ~1.1s per version)
- ✅ Added `--quiet` flag to pip install for cleaner output
- ✅ Venv validation to ensure correct Lance version is installed

## Configuration Options

### Persistent vs Temporary Venvs

By default, venvs are persistent. To use temporary venvs (old behavior):

```bash
COMPAT_TEMP_VENV=1 pytest tests/forward_compat/ --run-compat
```

### Cleaning Up Persistent Venvs

To remove all cached venvs:

```bash
rm -rf ~/.cache/lance-compat-venvs/
```

Or to remove specific versions:

```bash
rm -rf ~/.cache/lance-compat-venvs/venv_0.30.0
```

## Potential Future Optimizations

### High Impact

1. **Parallel venv creation** (saves ~8s × versions):
   - Create all venvs in parallel at session start
   - Most beneficial for CI or first-time setup
   - Requires refactoring VenvFactory

2. **Pre-built Docker image** (eliminates install time entirely):
   - Container with all Lance versions pre-installed
   - Good for CI, not for local dev

### Low Impact

1. **Venv with --without-pip**: Saves ~0.5s
   - Requires symlinking pip from parent venv
   - Adds complexity

## Recommendations

For local development:
- ✅ **Use persistent venvs** (default) - 5x speedup after first run
- Run tests frequently without worrying about setup time
- Manually clean cache if disk space is a concern

For CI:
- Consider caching `~/.cache/lance-compat-venvs/` across CI runs
- Or use `COMPAT_TEMP_VENV=1` for clean environments each time

## Performance Instrumentation

Detailed timing information is available by setting the `DEBUG` environment variable:

```bash
# Run tests with timing instrumentation
DEBUG=1 pytest tests/forward_compat/ --run-compat -v -s

# Normal run (clean output, no timing)
pytest tests/forward_compat/ --run-compat -v
```

The timing output shows:
- `[TIMING]` - Main process timing (venv creation, IPC)
- `[VENV TIMING]` - Subprocess timing (actual method execution)

This helps identify bottlenecks:
- If `receive` time is much larger than `[VENV TIMING]` execution, the bottleneck is Lance import
- If `[VENV TIMING]` is large, the bottleneck is the actual test logic
