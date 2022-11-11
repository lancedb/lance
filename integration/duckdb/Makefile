#

release-linux:
	docker run --rm -v ${CURDIR}:/code -w /code quay.io/pypa/manylinux2014_x86_64 \
		/bin/bash -c "cmake -B manylinux-build && cd manylinux-build && make -j $(JOBS)"

release-cuda:
	cd cuda && docker compose run cuda /code/cuda/build_lance.sh && cd ..

release-osx:
	cmake -B osx-build && make -C osx-build -j