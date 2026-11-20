"""验证暗台面、靠边细线、噪点与真实设计孔洞。"""

import numpy as np
from run import MODELS, Plane, predict

shape = (192, 256)
fabric = np.array([70, 65, 55, 61, 75, 84, 91, 64], np.uint8)
ink = np.array([185, 45, 165, 160, 180, 192, 55, 195], np.uint8)
scene = np.broadcast_to(fabric[:, None, None], (8, *shape)).copy()
scene[:, :, :18] = 9  # 与左边界相连的空台面
scene[:, 52:140, 80:172] = ink[:, None, None]
scene[:, 85:105, 108:135] = fabric[:, None, None]  # 设计内孔
scene[:, 110:120, 88:98] = 9  # 被图案包围的实心黑墨
scene[:, 32:35, 65:120] = ink[:, None, None]  # 真实细笔画
scene[:, 16, 48] = ink  # 单像素干扰
scene[:, 152:160, 188:196] = ink[:, None, None]
scene[:, 40:135, 19:22] = ink[:, None, None]  # 沿布边延伸的独立线条
scene[:, 126:129, 22:55] = ink[:, None, None]  # 横穿布边方向的真实笔画，与边缘伪线分离
scene[:, 124:131, 19:23] = fabric[:, None, None]
scene[:, 160:164, 72:76] = ink[:, None, None]  # 远离图案的十六像素孤点
planes = (Plane * 8)(
    *[
        Plane(a.ctypes.data, a.nbytes, shape[1], shape[0], a.strides[0], b)
        for b, a in enumerate(scene)
    ]
)
for route, t in [(8, 112), (4, 104)]:
    mask, _, r = predict(MODELS / f"rf{route}.yml", planes, shape, t)
    assert not mask[:, 0:15].any()
    assert not mask[89:101, 112:128].any()
    assert mask[113:118, 90:96].all()
    assert mask[33, 69:115].all() and mask[154:158, 190:194].all()
    assert not mask[50:116, 19:22].any() and not mask[160:164, 72:76].any()
    assert mask[127, 28:51].all()
    assert not mask[16, 48] and r.count == np.count_nonzero(mask)

    # 旧接口与可调接口的默认掩膜应逐像素相同。
    def tuned(settings):
        return predict(MODELS / f"rf{route}.yml", planes, shape, t, settings)[0]

    assert np.array_equal(mask, tuned((96, 80, 46)))
    assert tuned((0, 80, 46))[161, 73] and not tuned((512, 80, 46))[155, 191]
    # 这处内孔为 540 像素。只提高面积上限仍受证据约束，最大容差会将它填满。
    assert not tuned((96, 539, 255))[94, 119]
    assert not tuned((96, 540, 0))[94, 119]
    assert tuned((96, 540, 255))[94, 119] and not tuned((96, 0, 255))[94, 119]
    try:
        tuned((-1, 80, 46))
    except ValueError:
        pass
    else:
        raise AssertionError("DLL 应拒绝负面积")
print(
    "PASS: exterior, detached edge line, cross-edge stroke, small speck, enclosed dark ink, genuine hole and printed dot on both routes"
)
