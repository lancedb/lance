FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y \
    && apt-get -y clean \
	&& curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
