"""用非紧密步长和乱序输入验证已构建的 Release DLL。"""

import ctypes as C
import numpy as np
from run import dll, Plane, Result, MODELS, ptr, check

ctx = ptr()
check(dll.ms_load(str(MODELS / "rf8.yml"), C.byref(ctx)))
try:
    rng = np.random.default_rng(5)
    h, w = 64, 96
    rows = [rng.integers(50, 160, (h, w + 7), dtype=np.uint8) for _ in range(8)]
    planes = (Plane * 8)(
        *[Plane(a.ctypes.data, a.nbytes, w, h, w + 7, i) for i, a in enumerate(rows)]
    )
    mask = np.empty((h, w), np.uint8)
    score = np.empty_like(mask)
    r = Result()

    def call(p=planes, n=8, cap=w * h, cancel=None):
        return dll.ms_run(
            ctx, p, n, 128, mask.ctypes.data, score.ctypes.data, cap, C.byref(r), cancel
        )

    check(call())
    a = score.copy()
    assert set(np.unique(mask)) <= {0, 255} and (mask > 0).sum() == r.count
    check(call((Plane * 8)(*reversed(planes))))
    assert np.array_equal(a, score)
    assert call(n=7) != 0 and call(cap=w * h - 1) != 0
    planes[1].band = 0
    assert call() != 0
    planes[1].band = 1
    planes[2].h = h - 1
    assert call() != 0
    planes[2].h = h
    planes[3].size = 5
    assert call() != 0
    planes[3].size = rows[3].nbytes
    stop = C.c_int(1)
    assert call(cancel=C.byref(stop)) != 0 and dll.ms_error() == b"Cancelled"
    check(call())
    assert np.array_equal(a, score)
    assert dll.ms_load("missing-model.yml", C.byref(ptr())) != 0
    print(
        "PASS: stride, permutation, mask/score, missing band, duplicate band, dimensions, capacities, cancellation, retry, missing model"
    )
finally:
    dll.ms_free(ctx)
