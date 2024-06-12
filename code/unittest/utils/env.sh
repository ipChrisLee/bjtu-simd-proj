#!/usr/bin/env bash

if [ ! -f "unittest/utils/anchor" ] || [ "$(head -1 unittest/utils/anchor)" != "unittest/utils | unittest" ]; then echo "Anchor [unittest/utils] not found!"; exit 1; fi

# utils/build-debug-dev.sh
utils/build-debug.sh

# export PATH="$PATH:$(pwd)/build-debug-dev"
export PATH="$PATH:$(pwd)/build-debug"