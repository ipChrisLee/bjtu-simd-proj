# code

## Before all

You SHOULD run following commands under root of code inside container.

## How to build

```bash
utils/build-debug-dev.sh # for qemu aarch64 elf compile.
utils/build-debug.sh # for host elf compile.
```

Build is written to `build-debug-dev` or `build-debug`.

## How to test

```bash
unittest/utils/gen-suite.sh
```

Test suite is written to `unittest/test-suite`.

## How to run test

```bash
unittest/utils/run-test-of-layer.sh "relu"
```

This will test layer `relu`. You can replace `relu` with some other layer name to run other layer tests.