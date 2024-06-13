#!/usr/bin/env bash

if [ ! -f "utils/anchor" ] || [ "$(head -1 utils/anchor)" != "utils | bjtu-simd-dev-code" ]; then echo "Anchor [utils] not found!"; exit 1; fi

source utils/env.sh

echo "Running \`alex-net-std 1 42 \"$alexNetSimdWorkspace\"\`"
alex-net-simd 1 42 "$alexNetSimdWorkspace"
