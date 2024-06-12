#!/usr/bin/env python3

import lib
from pathlib import Path
import sys
import torch


def gen_softmax(tInfoPath: Path):
    print(tInfoPath)
    srcShape = lib.gen_random_dim_array(l=2, r=5)
    src = lib.gen_uniform_tensor(srcShape, (-1.0, 1.0))
    axis = lib.gen_rand_dim()
    goldenDst = torch.softmax(input=torch.tensor(src), dim=axis)
    lib.dump(tInfoPath=tInfoPath, src=src, goldenDst=goldenDst,
             axis=axis, atol=1e-3, rtol=1e-2)


def gen_simple_softmax(tInfoPath: Path):
    print(tInfoPath)
    srcShape = lib.gen_random_dim_array(d=2, l=2, r=3)
    src = lib.gen_uniform_tensor(srcShape, (-1.0, 1.0))
    axis = lib.gen_rand_dim(d=2)
    goldenDst = torch.softmax(input=torch.tensor(src), dim=axis)
    lib.dump(tInfoPath=tInfoPath, src=src, goldenDst=goldenDst,
             axis=axis, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    gen_simple_softmax(Path(sys.argv[1]))
