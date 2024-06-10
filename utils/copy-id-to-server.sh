#!/usr/bin/env bash

if [ ! -f "./anchor" ] || [ "$(head -1 ./anchor)" != ". | bjtu-simd-dev" ]; then echo "Anchor [.] not found!"; exit 1; fi

source env.sh

if ! [ -f "${HOME}/.ssh/id_ed25519.pub" ]; then
	echo "Do not find ~/.ssh/id_ed25519.pub, please generate one key."
	exit 1
fi

if [ -n "$proxyJump" ]; then
	ssh-copy-id -i ~/.ssh/id_ed25519.pub "$proxyJump"
fi


if [ -z "$proxyJump" ]; then
	ssh-copy-id -i ~/.ssh/id_ed25519.pub "$remote"
else
	ssh-copy-id -o "ProxyJump=$proxyJump" -i ~/.ssh/id_ed25519.pub "$remote"
fi

