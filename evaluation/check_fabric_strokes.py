"""验证同一弱光谱对比度下，细长连续笔画和紧凑孤点走不同的去噪判据。"""

import ctypes as C

import numpy as np

from run import Plane, Result, check, dll, ptr


def main():
    rng = np.random.default_rng(8)
    width = height = 384
    # 与复测使用相同的固定布纹样本。两个形状只在第一个波段有相同的灰度变化。
    bands = []
    for _ in range(8):
        bands = [
            np.clip(rng.normal(92 + b * 7, 2.5, (height, width)), 0, 255).astype(np.uint8)
            for b in range(8)
        ]
    line = np.s_[160:163, 160:182]
    dot = np.s_[176:179, 160:163]
    bands[0][line] += 19
    bands[0][dot] += 19
    planes = (Plane * 8)(*[
        Plane(image.ctypes.data, image.nbytes, width, height, width, band)
        for band, image in enumerate(bands)
    ])
    dll.ms_run_fabric_configured.argtypes = [
        C.POINTER(Plane), C.c_int, C.c_uint8, ptr, ptr, C.c_int64,
        C.POINTER(Result), ptr, ptr, C.c_int, C.c_int,
    ]
    mask = np.empty((height, width), np.uint8)
    score = np.empty_like(mask)
    report = Result()
    args = (planes, 8, 128, mask.ctypes.data, score.ctypes.data, mask.size, C.byref(report), None, None)
    check(dll.ms_run_fabric_configured(*args, 0, 80))
    assert np.count_nonzero(mask[dot]) > 0
    assert 128 <= score[line].max() < 208
    before = mask[line].copy()

    check(dll.ms_run_fabric_configured(*args, 128, 80))
    assert np.count_nonzero(mask[line]) >= 30
    np.testing.assert_array_equal(mask[line], before)
    assert not mask[dot].any()
    print("通过：原弱笔画保留，紧凑孤点被过滤。")


if __name__ == "__main__":
    main()
