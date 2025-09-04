#!/bin/bash
set -e

# Strip debug symbols from wheel files
for wheel in "$@"; do
    echo "Processing $wheel"
    
    # Get absolute path to avoid issues with relative paths
    WHEEL_PATH=$(realpath "$wheel")
    
    # Create temp directory
    TEMP_DIR=$(mktemp -d)
    
    # Extract wheel
    unzip -q "$WHEEL_PATH" -d "$TEMP_DIR"
    
    # Strip all .so files
    find "$TEMP_DIR" -name "*.so" -type f | while read -r so_file; do
        echo "  Stripping: $(basename "$so_file")"
        strip --strip-debug "$so_file"
    done
    
    # Remove original wheel and repack
    rm "$WHEEL_PATH"
    (cd "$TEMP_DIR" && zip -qr "$WHEEL_PATH" .)
    
    # Cleanup
    rm -rf "$TEMP_DIR"
    
    echo "  Repacked: $(basename "$WHEEL_PATH")"
done