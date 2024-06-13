if [ -z "$envSourced" ]; then

export envSourced=y

if [ ! -f "utils/anchor" ] || [ "$(head -1 utils/anchor)" != "utils | bjtu-simd-dev-code" ]; then echo "Anchor [utils] not found!"; exit 1; fi

if [ -f "${HOME}/.local/bin/cmake" ]; then
    export PATH="${HOME}/.local/bin:$PATH"
fi

if [ -n "$onQemu" ]; then
    utils/build-debug-dev.sh
    export PATH="$PATH:$(pwd)/build-debug-dev"
elif [ -n "$onArm" ]; then
    utils/build-debug.sh
    export PATH="$PATH:$(pwd)/build-debug"
elif [ -n "$onRel" ]; then
    utils/build-release.sh
    export PATH="$PATH:$(pwd)/build-release"
else
    utils/build-debug.sh
    export PATH="$PATH:$(pwd)/build-debug"
fi

export alexNetStdWorkspace="$(pwd)/workspace-alex-net/std"
export alexNetSimdWorkspace="$(pwd)/workspace-alex-net/simd"
mkdir -p "$alexNetStdWorkspace"
mkdir -p "$alexNetSimdWorkspace"

fi # if [ -z "$envSourced" ]; then

# if (onQemu)
#   build: "-dev" with toolchain.cmake
#   enable: simd && std
#   run: on qemu
# else
#   if (onArm)
#     build: without toolchain.cmake
#     enable: simd && std
#     run: on arm host
#   else if(onRel)
#     build: "-release" without toolchain.cmake
#     enable: simd && std
#     run: on arm host and release version
#   else
#     build: without toolchain.cmake
#     enable: std
#     run: on host (maybe amd64)