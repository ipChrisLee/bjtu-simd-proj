if [ ! -f "utils/anchor" ] || [ "$(head -1 utils/anchor)" != "utils | bjtu-simd-dev-code" ]; then echo "Anchor [utils] not found!"; exit 1; fi

if [ -f "${HOME}/.local/bin/cmake" ]; then
    export PATH="${HOME}/.local/bin:$PATH"
fi

if [ -n "$onQemu" ]; then
    utils/build-debug-dev.sh
    export PATH="$PATH:$(pwd)/build-debug-dev"
else
    utils/build-debug.sh
    export PATH="$PATH:$(pwd)/build-debug"
fi


# if (onQemu)
#   build: "-dev" with toolchain.cmake
#   enable: simd && std
#   run: on qemu
# else
#   if (onArm)
#     build: without toolchain.cmake
#     enable: simd && std
#     run: on arm host
#   else
#     build: without toolchain.cmake
#     enable: std
#     run: on host (maybe amd64)