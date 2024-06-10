if [ ! -f "utils/anchor" ] || [ "$(head -1 utils/anchor)" != "utils | bjtu-simd-dev-code" ]; then echo "Anchor [utils] not found!"; exit 1; fi

if [ -f "${HOME}/.local/bin/cmake" ]; then
    export PATH="${HOME}/.local/bin:$PATH"
fi