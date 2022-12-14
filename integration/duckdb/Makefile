#

UID := $(shell id -u)
GID := $(shell id -g)

release-linux:
	docker build -t duckdb_manylinux -f tools/Dockerfile.manylinux2014 tools
	docker run --rm \
		-v ${CURDIR}/../..:/code \
		-w /code/integration/duckdb \
		--user ${UID}:${GID} \
		duckdb_manylinux \
		/bin/bash -c "cmake -DLANCE_BUILD_PYTORCH=FALSE -B manylinux-build && cd manylinux-build && make -j $(JOBS)"

debug-manylinux:
	docker build -t duckdb_manylinux -f tools/Dockerfile.manylinux2014 tools
	docker run --rm -it \
		-v ${CURDIR}/../..:/code \
		-w /code/integration/duckdb \
		--user ${UID}:${GID} \
		duckdb_manylinux \
		/bin/bash

release-cuda:
	cd cuda && docker compose run cuda /code/cuda/build_lance.sh && cd ..

release-osx:
	cmake -B osx-build && make -C osx-build -j
