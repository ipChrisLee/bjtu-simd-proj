FROM ubuntu:22.04

RUN mkdir -p /workspace

WORKDIR /workspace

# ------------ install necessary tools ------------
RUN apt-get update
RUN apt-get install -y clang python3 python3-pip clangd
RUN pip3 install cmake

