#!/bin/bash
set -e

# Strip debug symbols from wheel files
for wheel in "$@"; do
    echo "Processing $wheel"
    
    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    
    # Extract wheel
    unzip -q "$wheel" -d "$TEMP_DIR"
    
    # Strip all .so files
    find "$TEMP_DIR" -name "*.so" -type f | while read -r so_file; do
        echo "  Stripping: $(basename "$so_file")"
        strip --strip-debug "$so_file"
    done
    
    # Repack wheel
    rm "$wheel"
    (cd "$TEMP_DIR" && zip -qr "$wheel" .)
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    
    echo "  Repacked: $(basename "$wheel")"
done