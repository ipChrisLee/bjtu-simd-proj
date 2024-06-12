FROM ubuntu:22.04

RUN mkdir -p /workspace

WORKDIR /workspace

# ------------ install necessary toolchain ------------
RUN apt-get update
RUN apt-get install -y clang clangd gcc-aarch64-linux-gnu lld
RUN apt-get install -y python3 python3-pip
RUN pip3 install cmake
RUN apt-get install -y gdb

# ------------ install qemu for test ------------
RUN apt-get install -y qemu qemu-system-arm qemu-efi-aarch64 qemu-efi
