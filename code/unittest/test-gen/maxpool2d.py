#!/usr/bin/env python3

import lib
from pathlib import Path
import sys
import torch


def gen_maxpool2d(tInfoPath: Path):
    print(tInfoPath)
    srcShape = lib.gen_random_dim_array(4, l=10, r=15)
    src = lib.gen_uniform_tensor(srcShape, (-1.0, 1.0))
    kernelSize = lib.gen_random_dim_array(2, l=4, r=6)
    padding = lib.gen_random_dim_array(2, l=0, r=3)
    if lib.gen_uniform_bool():
        stride = lib.gen_random_dim_array(2, l=1, r=4)
    else:
        stride = [1, 1]
    goldenDst = torch.max_pool2d(input=torch.tensor(
        src), kernel_size=kernelSize, padding=padding, stride=stride).numpy()
    lib.dump(tInfoPath=tInfoPath, src=src, kernelSize=kernelSize,
             padding=padding, stride=stride, goldenDst=goldenDst)


if __name__ == "__main__":
    gen_maxpool2d(Path(sys.argv[1]))
