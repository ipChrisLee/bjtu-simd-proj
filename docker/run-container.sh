#!/usr/bin/env bash

if [ ! -f "./anchor" ] || [ "$(head -1 ./anchor)" != ". | bjtu-simd-dev" ]; then echo "Anchor [.] not found!"; exit 1; fi

source env.sh

docker run \
	--name "${containerName}" \
	-h "${containerName}" \
	"$@" \
	-itd "${containerName}_image"
