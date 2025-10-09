Tests for memory and IO usage.

## Debugging memory usage

Once you've identified a test that is using too much memory, you can use
bytehound to find the source of the memory usage.

```shell
LD_PRELOAD=/usr/local/lib/libbytehound.so \
    RUST_ALLOC_TIMINGS=true \
    cargo test --test resource_tests resource_test::index::test_label_list_lifecycle
bytehound server memory-profiling_*.dat
```