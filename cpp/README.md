# Lance: A Columnar Data Format


# Development

## Code Style

The core of `Lance` data format is developed in `C++20`.

The code base follows Apache Arrow and [Google C++ Code Style](https://google.github.io/styleguide/cppguide.html).
With exceptions:
* Line width: 100

The code style is enforced via [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html). Please make sure `clang-format` has been set up appropriately in the IDE.

## Build

The code is developed on `macOS 12+` and a recent Linux (`Ubuntu 22.04`). Please file bugs or contribute fixes if you encounter issues with other Linux flavors.

A recent C++ compiler that supports C++20 is needed. The toolchain that has been tested is:

* AppleClang >= 13.1.6
* GCC >= 11.2
* CMake >= 3.22
* Apache Arrow >= 8.0
* Protobuf > 3.12

```sh
# On macOS 12+
brew install apache-arrow cmake protobuf
# Optionally, build document
brew install doxygen
```

```sh
# On Ubuntu 22.04
# Step 1. Follow https://arrow.apache.org/install/

# Step 2. Install arrow, protobuf, cmake
sudo apt install -y -V libarrow-dev libarrow-dataset-dev libparquet-dev libarrow-python-dev cmake libprotobuf-dev
# Optionally, build document
sudo apt install -y -V install doxygen
```

Once the installation is completed, we can check out `lance` from github and start building.

```sh
git clone git@github.com:eto-ai/lance.git
cd lance/cpp
mkdir -p build
cmake -B build
cd build
make -j

# Run unit tests
make test
```
