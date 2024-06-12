#!/usr/bin/env python3

from pathlib import Path
import sys
import typing as typ
import numpy as np
import os
from relu import gen_relu
from softmax import gen_softmax


def gen_dispatch(layer: str, testPath: Path):
    match layer:
        case "relu":
            gen_relu(testPath)
        case "softmax":
            gen_softmax(testPath)
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
    layerList = ["relu", "softmax"]
    countEveryLayer = 5
    gen(suitePath=suitePath, layerList=layerList, countEveryLayer=countEveryLayer)
