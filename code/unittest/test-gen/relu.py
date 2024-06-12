#!/usr/bin/env python3

import lib
from pathlib import Path
import sys
import torch


def gen_relu(tInfoPath: Path):
    print(tInfoPath)
    srcShape = lib.gen_random_dim_array()
    src = lib.gen_uniform_tensor(srcShape, (-1.0, 1.0))
    goldenDst = torch.relu(torch.tensor(src)).numpy()
    lib.dump(tInfoPath=tInfoPath, src=src, goldenDst=goldenDst)


if __name__ == "__main__":
    gen_relu(Path(sys.argv[1]))
