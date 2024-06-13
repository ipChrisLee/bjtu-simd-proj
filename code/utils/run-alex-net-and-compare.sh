#!/usr/bin/env bash

if [ ! -f "utils/anchor" ] || [ "$(head -1 utils/anchor)" != "utils | bjtu-simd-dev-code" ]; then echo "Anchor [utils] not found!"; exit 1; fi

source utils/env.sh

utils/run-alex-net-std.sh
utils/run-alex-net-simd.sh

tensor-diff "$alexNetSimdWorkspace/output.txt" "$alexNetStdWorkspace/output.txt" "1e-5" "1e-3"