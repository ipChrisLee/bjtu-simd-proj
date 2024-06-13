#!/usr/bin/env python3

import lib
from pathlib import Path
import sys
import torch


def gen_conv2d(tInfoPath: Path):
    print(tInfoPath)
    srcShape = lib.gen_random_dim_array(l=4, r=5)
    kerShape = lib.gen_random_dim_array(d=4, l=2, r=4)
    kerShape[1] = srcShape[1]
    src = lib.gen_uniform_tensor(srcShape, (0, 1.0))
    kernel = lib.gen_uniform_tensor(kerShape, (0, 1.0))
    goldenDst = torch.conv2d(input=torch.tensor(src), weight=torch.tensor(kernel))
    lib.dump(tInfoPath=tInfoPath, src=src, goldenDst=goldenDst,
             kernel=kernel, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    gen_conv2d(Path(sys.argv[1]))
