#!/usr/bin/env bash

if [ ! -f "unittest/utils/anchor" ] || [ "$(head -1 unittest/utils/anchor)" != "unittest/utils | unittest" ]; then echo "Anchor [unittest/utils] not found!"; exit 1; fi

source utils/env.sh

export suitePath="$(pwd)/unittest/test-suite"