# Lance: A Columnar File Format


## Development

The core of `Lance` data format is developed in `C++20`.

The code base follows Apache Arrow and [Google C++ Code Style](https://google.github.io/styleguide/cppguide.html).
With exceptions:
* Line width: 100

The code style is enforced via [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html). Please make sure `clang-format` has been set up appropriately in the IDE.

The code is developed on `macOS 12+` and recent Linux (`Ubuntu 22.04`). Feel free to file bugs or contribute if you encounter issues with other Linux flavors.

The C++ codebase is on C++20, so a recent compiler is expected, the toolchain that is tested:

* AppleClang >= 13.1.6
* GCC >= 11.2
* CMake >= 3.22
* Apache Arrow >= 8.0
* Protobuf > 3.12

```sh
# On macOS 12+
brew install apache-arrow cmake protobuf
```

```sh
# On Ubuntu 22.04
# Follow https://arrow.apache.org/install/
sudo apt update
sudo apt install -y -V ca-certificates lsb-release wget
wget https://apache.jfrog.io/artifactory/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt install -y -V ./apache-arrow-apt-source-latest-$(lsb_release --codename --short).deb
sudo apt update

# Install arrow, protobuf, cmake
sudo apt install -y -V libarrow-dev libarrow-dataset-dev libparquet-dev libarrow-python-dev cmake libprotobuf-dev protobuf-compiler
```

Once the installation is completed, we can check out `lance` from github and start building.

```sh

git clone git@github.com:eto-ai/lance.git
cd lance/cpp
mkdir -p build
cmake -B build
cd build
make -j

# Run unit test
make test
```
