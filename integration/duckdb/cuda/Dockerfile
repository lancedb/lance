FROM pytorch/manylinux-cuda117 AS builder

RUN rm -f /usr/local/cuda && ln -s /usr/local/cuda-11.7 /usr/local/cuda

# Upgrade cmake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.24.2/cmake-3.24.2-linux-x86_64.sh && \
    mkdir -p /usr/local/cmake && \
    sh cmake-3.24.2-linux-x86_64.sh --skip-license --prefix=/usr/local/cmake

RUN yum install -y devtoolset-11-gcc-c++
