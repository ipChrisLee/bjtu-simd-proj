#!/usr/bin/env bash

if [ ! -f "unittest/utils/anchor" ] || [ "$(head -1 unittest/utils/anchor)" != "unittest/utils | unittest" ]; then echo "Anchor [unittest/utils] not found!"; exit 1; fi

export onQemu="y"

source unittest/utils/env.sh

export layer="$1"

echo "Running for layer {$layer}"

find "$suitePath/$layer" -type f | while read -r tInfoPath; do
	echo "Running: \`run-layer-test-simd \"$layer\" \"$tInfoPath\"\`"
	run-layer-test-simd "$layer" "$tInfoPath"
done