#!/usr/bin/env bash

if [ ! -f "./anchor" ] || [ "$(head -1 ./anchor)" != ". | bjtu-simd-dev" ]; then echo "Anchor [.] not found!"; exit 1; fi

source secret.sh

function check_env_var() {
	if [ -z "$(eval echo "\$$1")" ]; then
		echo "Require env var \"$1\" in secret.sh"
		exit 1
	fi
}

check_env_var remoteServerUser
check_env_var remoteServerAddr

if [ -z "$sshCommand" ]; then
	export sshCommand="ssh"
fi

if [ -z "$containerName" ]; then
	export containerName="bjtu-simd-dev"
fi
