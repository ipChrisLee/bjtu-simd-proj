#!/usr/bin/env bash

if [ ! -f "./anchor" ] || [ "$(head -1 ./anchor)" != ". | bjtu-simd-dev" ]; then echo "Anchor [.] not found!"; exit 1; fi

source env.sh

true_ssh 'rm -rf code'
true_scp -r code "$remote":~/code
true_ssh 'cd code; rm -rf build-debug; utils/build-debug.sh'
true_ssh 'cd code; build-debug/alex-net'
