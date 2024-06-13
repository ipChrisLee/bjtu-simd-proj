#!/usr/bin/env bash

if [ ! -f "utils/anchor" ] || [ "$(head -1 utils/anchor)" != "utils | bjtu-simd-dev-code" ]; then echo "Anchor [utils] not found!"; exit 1; fi

mkdir -p build-release
cmake --preset build-release-conf
cmake --build build-release