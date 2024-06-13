#!/usr/bin/env python3

import lib
from pathlib import Path
import sys
import numpy as np
import torch


def gen_fc(tInfoPath: Path):
    print(tInfoPath)
    srcShape = lib.gen_random_dim_array(d=4, l=30, r=35)
    weightShape = lib.gen_random_dim_array(d=2, l=30, r=35)
    weightShape[1] = srcShape[-1]
    src = lib.gen_uniform_tensor(srcShape, (0, 1.0))
    weight = lib.gen_uniform_tensor(weightShape, (0, 1.0))
    goldenDst = torch.matmul(torch.tensor(
        src), torch.tensor(np.transpose(weight)))
    lib.dump(tInfoPath=tInfoPath, src=src, goldenDst=goldenDst,
             weight=weight, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    gen_fc(Path(sys.argv[1]))
