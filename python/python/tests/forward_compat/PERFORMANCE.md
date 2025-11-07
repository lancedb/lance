# Compatibility Test Performance Analysis

## Timing Breakdown (per version, first test)

```
Total: ~16-17s
├── Virtual environment setup: ~7-8s (47%)
│   ├── venv creation: 2.2s
│   └── package install (pylance + pytest): 4.9s
├── First Lance import in subprocess: ~5.0s (29%)
├── Test execution: ~0.04s (0.2%)
└── Overhead (pytest, data creation, etc.): ~4s (24%)
```

## What's Working Well

1. **Persistent subprocess**: Subsequent method calls are 500x faster (5s → 0.01s)
2. **Venv reuse**: Virtual environments are cached across tests in same session
3. **Pip cache**: Leveraging ~11GB pip cache for faster installs

## Current Optimizations

- ✅ Removed pip upgrade step (saves ~1.1s per version)
- ✅ Added `--quiet` flag to pip install for cleaner output

## Potential Future Optimizations

### High Impact (but complex)

1. **Parallel venv creation** (saves ~8s × versions):
   - Create all venvs in parallel at session start
   - Requires refactoring VenvFactory to pre-create venvs

2. **Persistent venv directory** (saves ~8s on subsequent runs):
   - Store venvs outside /tmp to persist across pytest sessions
   - Add venv version/integrity checking
   - Cleanup strategy for old venvs

3. **Pre-built Docker image** (eliminates install time entirely):
   - Container with all Lance versions pre-installed
   - Good for CI, not for local dev

### Low Impact

1. **Venv with --without-pip**: Saves ~0.5s
   - Requires symlinking pip from parent venv
   - Adds complexity

2. **Lazy Lance import**: Not applicable
   - Import happens on first method call (already optimized)

## Recommendations

For local development:
- Current setup is good - optimizations have diminishing returns
- Most time is in package installation (pip already optimized)

For CI:
- Consider parallel venv creation if testing many versions
- Consider persistent venv cache across CI runs

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
