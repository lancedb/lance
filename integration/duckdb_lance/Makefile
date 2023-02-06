#

BUILD_FLAGS=-DEXTENSION_STATIC_BUILD=1

# Debug build
build:
	mkdir -p build/debug && \
	cd build/debug && \
	cmake $(GENERATOR) $(FORCE_COLOR) -DCMAKE_BUILD_TYPE=Debug ${BUILD_FLAGS} ../../duckdb/CMakeLists.txt -DEXTERNAL_EXTENSION_DIRECTORIES=../../duckdb_lance -B. && \
	cmake --build . --config Debug
.PHONY: build
