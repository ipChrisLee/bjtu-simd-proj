#!/usr/bin/env bash

if [ ! -f "utils/anchor" ] || [ "$(head -1 utils/anchor)" != "utils | bjtu-simd-dev-code" ]; then echo "Anchor [utils] not found!"; exit 1; fi

source utils/env.sh

export workspace="$(pwd)/workspace-alex-net/std"

mkdir -p "$workspace"

echo "Running \`alex-net-std 1 42 \"$workspace\"\`"
alex-net-std 1 42 "$workspace"
