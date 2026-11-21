"""逐像素比较全图快速投票与 OpenCV getVotes，包括非整块长度和兼容回退。"""

import ctypes as C
import tempfile
from pathlib import Path

import numpy as np

from run import MODELS, Plane, check, dll, predict, ptr


def compare(path):
    # 尺寸低于配准窗口要求，所有通道使用零位移；宽度和总点数均不对齐到 16。
    h, w = 31, 47
    rng = np.random.default_rng(712)
    images = rng.integers(0, 256, (8, h, w + 5), dtype=np.uint8)
    planes = (Plane * 8)(
        *[Plane(a.ctypes.data, a.nbytes, w, h, a.strides[0], b) for b, a in enumerate(images)]
    )
    xy = np.indices((h, w), dtype=np.int32)[::-1].reshape(2, -1).T.copy()
    features = np.empty((w * h, 10), np.float32)
    check(dll.ms_sample_mode(planes, 8, xy.ctypes.data, len(xy), features.ctypes.data, 1))
    reference = np.empty(w * h, np.uint8)
    handle = ptr()
    check(dll.ms_load(str(path), C.byref(handle)))
    try:
        check(dll.ms_score_rows(handle, features.ctypes.data, len(xy), reference.ctypes.data))
    finally:
        dll.ms_free(handle)
    _, scores, _ = predict(path, planes, (h, w), 112)
    np.testing.assert_array_equal(scores.ravel(), reference)


if __name__ == "__main__":
    for route in (2, 3, 4, 8):
        compare(MODELS / f"rf{route}.yml")
    with tempfile.TemporaryDirectory() as folder:
        fallback = Path(folder) / "inverse.yml"
        original = (MODELS / "rf8.yml").read_text()
        assert "le:" in original
        fallback.write_text(original.replace("le:", "gt:", 1))
        compare(fallback)
    print("通过：四种特征配置、条带尾部、行填充和反向分裂回退均与 OpenCV 树票一致。")
