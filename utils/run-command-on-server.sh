#!/usr/bin/env bash

if [ ! -f "./anchor" ] || [ "$(head -1 ./anchor)" != ". | bjtu-simd-dev" ]; then echo "Anchor [.] not found!"; exit 1; fi

source env.sh

sudo rm -rf /tmp/code
sudo cp -r code /tmp/code
sudo find /tmp/code -type d -name "build*" -exec rm -rf {} +
sudo find /tmp/code -type f -name "tInfo*" -exec rm {} +
sudo find /tmp/code -type d -name "workspace*" -exec rm -rf {} +
true_ssh 'rm -rf ~/code || true'
true_scp -r /tmp/code "$remote":~/code
# to timeit, add `export doTimeit=y;`
true_ssh "cd ~/code; $1"
