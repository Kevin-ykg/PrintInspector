"""验证带正负位移的八帧、非顺序输入以及 W 坐标导出。"""

import ctypes as C
import numpy as np
from run import MODELS, Plane, Result, check, dll, ptr

rng = np.random.default_rng(123)
w, h = 384, 320
white = rng.integers(30, 205, (h, w), np.uint8)
offsets = [(2, -3), (-2, 3), (1, 1), (0, -1), (0, 2), (3, 0), (-1, 0), (0, 0)]
images = [np.roll(white, (dy, dx), axis=(0, 1)) for dx, dy in offsets]
planes = (Plane * 8)(*[Plane(images[b].ctypes.data, w * h, w, h, w, b) for b in reversed(range(8))])
xy = (C.c_int * 16)()
m, s = np.zeros_like(white), np.zeros_like(white)
r, model = Result(), ptr()
dll.ms_run_registered.argtypes = dll.ms_run_configured.argtypes + [C.POINTER(C.c_int)]
check(dll.ms_load(str(MODELS / "rf8.yml"), C.byref(model)))
try:
    args = (
        model,
        planes,
        8,
        112,
        m.ctypes.data,
        s.ctypes.data,
        m.size,
        C.byref(r),
        None,
        96,
        80,
        46,
    )
    check(dll.ms_run_registered(*args, xy))
    np.testing.assert_array_equal(np.array(xy).reshape(8, 2), offsets)
    assert not (r.flags & 1)
    m1, s1 = m.copy(), s.copy()
    check(dll.ms_run_configured(*args))
    np.testing.assert_array_equal(m, m1)
    np.testing.assert_array_equal(s, s1)
finally:
    dll.ms_free(model)
print("通过：位移按波段编号返回，正负号与已知平移一致，新旧入口逐像素一致。")
