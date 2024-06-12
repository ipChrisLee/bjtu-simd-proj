from io import TextIOWrapper
import numpy as np
import typing as typ
from pathlib import Path


def _write_src(fp: TextIOWrapper, src: np.ndarray | None):
    if src is None:
        return
    fp.write("src ")
    D = len(src.shape)
    fp.write(f"{D} ")
    shape = src.shape
    for l in shape:
        fp.write(f"{l} ")
    lst = src.flatten()
    for l in lst:
        fp.write(f"{l} ")
    fp.write("\n")


def _write_goldenDst(fp: TextIOWrapper, goldenDst: np.ndarray | None):
    if goldenDst is None:
        return
    fp.write("goldenDst ")
    D = len(goldenDst.shape)
    fp.write(f"{D} ")
    shape = goldenDst.shape
    for l in shape:
        fp.write(f"{l} ")
    lst = goldenDst.flatten()
    for l in lst:
        fp.write(f"{l} ")
    fp.write("\n")


def _write_axis(fp: TextIOWrapper, axis: int | None = None):
    if axis is None:
        return
    fp.write(f"axis {axis} \n")


def _write_tol(fp: TextIOWrapper, tolName: str, tol: int | None = None):
    if tol is None:
        return
    fp.write(f"{tolName} {tol} \n")


def dump(
        tInfoPath: Path,
        src: np.ndarray | None = None,
        goldenDst: np.ndarray | None = None,
        axis: int | None = None,
        rtol: float | None = None,
        atol: float | None = None
):
    fp = open(tInfoPath, "w")
    _write_src(fp, src)
    _write_goldenDst(fp, goldenDst)
    _write_axis(fp, axis)
    _write_tol(fp, "rtol", rtol)
    _write_tol(fp, "atol", atol)
    fp.write("end\n")
    fp.close()


def gen_random_dim_array(d: int = 4, l: int = 1, r: int = 5) -> typ.List[int]:
    """generate array with d elements, value in range [l, r)
    """
    return np.random.randint(l, r, (d)).tolist()


def gen_rand_dim(d: int = 4) -> int:
    return np.random.randint(0, d)


def gen_rand_tensor(shape: typ.List[int]) -> np.array:
    return np.random.rand(*shape)


def gen_uniform_tensor(shape: typ.List[int], range: typ.Tuple[float, float]) -> np.array:
    return np.random.uniform(range[0], range[1], shape)
