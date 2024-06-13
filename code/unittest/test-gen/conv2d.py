#!/usr/bin/env python3

import lib
from pathlib import Path
import sys
import torch


def gen_conv2d(tInfoPath: Path):
    print(tInfoPath)
    srcShape = lib.gen_random_dim_array(d=4, l=30, r=35)
    kerShape = lib.gen_random_dim_array(d=4, l=10, r=15)
    kerShape[1] = srcShape[1]
    src = lib.gen_uniform_tensor(srcShape, (0, 1.0))
    kernel = lib.gen_uniform_tensor(kerShape, (0, 1.0))
    stride = lib.gen_random_dim_array(d=2, l=1, r=5)
    padding = lib.gen_random_dim_array(d=2, l=1, r=5)
    goldenDst = torch.conv2d(input=torch.tensor(src),
                             weight=torch.tensor(kernel), padding=padding, stride=stride)
    lib.dump(tInfoPath=tInfoPath, src=src, goldenDst=goldenDst,
             kernel=kernel, stride=stride, padding=padding, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    gen_conv2d(Path(sys.argv[1]))
