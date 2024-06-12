#!/usr/bin/env bash

if [ ! -f "unittest/utils/anchor" ] || [ "$(head -1 unittest/utils/anchor)" != "unittest/utils | unittest" ]; then echo "Anchor [unittest/utils] not found!"; exit 1; fi

source unittest/utils/env.sh

export layer="$1"

echo "Running for layer {$layer}"

find "unittest/$layer" -type f | while read -r tInfoPath; do
	echo "Running: \`run-layer-test-std \"$layer\" \"$tInfoPath\"\`"
	run-layer-test-std "$layer" "$tInfoPath"
done