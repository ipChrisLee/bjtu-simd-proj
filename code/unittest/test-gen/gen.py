#!/usr/bin/env python3

from pathlib import Path
import sys
import typing as typ
from relu import gen_relu
import numpy as np
import os


def gen_dispatch(layer: str, testPath: Path):
    match layer:
        case "relu":
            gen_relu(testPath)
            pass
        case _:
            raise Exception(f"Unkown layer {layer}")


def gen(suitePath: Path, layerList: typ.List[str], countEveryLayer: int):
    print(f"Gen suite {suitePath}")
    os.makedirs(suitePath, exist_ok=True)
    for layer in layerList:
        layerTestPath = suitePath / layer
        os.makedirs(layerTestPath, exist_ok=True)
        for i in range(countEveryLayer):
            testPath = layerTestPath / f"tInfo-{i}.txt"
            gen_dispatch(layer=layer, testPath=testPath)


if __name__ == "__main__":
    np.random.seed(42)
    suitePath = Path(sys.argv[1])
    layerList = ["relu"]
    countEveryLayer = 5
    gen(suitePath=suitePath, layerList=layerList, countEveryLayer=countEveryLayer)
