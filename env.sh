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

if [ -z "$containerName" ]; then
	export containerName="bjtu-simd-dev"
fi

export remote="$remoteServerUser@$remoteServerAddr"

function true_ssh() {
	if [ -z "$proxyJump" ]; then
		ssh "$remote" "$@"
	else
		ssh -o "ProxyJump=$proxyJump" "$remote" "$@"
	fi
}

function true_scp() {
	if [ -z "$proxyJump" ]; then
		scp "$remote" "$@"
	else
		scp -o "ProxyJump=$proxyJump" "$@"
	fi
}
