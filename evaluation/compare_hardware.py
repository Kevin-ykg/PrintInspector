"""用两套 Release DLL 复测新硬件 1～5 组，同一图像比较四张原分辨率掩膜。"""

import csv
import ctypes as C
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from run import BANDS, DATA, GROUPS, ROOT, Plane, Result, check, dll, group, labels, predict, ptr, sample_folder
from review import TABLE, crop_region, measure

WORKBENCH = ROOT.parent / "多光谱/多光谱烫印图案提取工作台"
OUT = ROOT / "results/跨硬件"
# 复核点绑定采集帧。若现场再次重排编号，拒绝将旧点位套在其他衣服上。
WHITE_SHA256 = {
    1: "35044a486601224cc5e2c952737e86b7b21ae858079d5a6d57ca0f27f101fe0d",
    2: "631cf988d69e385e219384f67e59d6322c1a66f263a4a9fa314be2d3a1b44fad",
    3: "1d9e36e646dbe08d9cfe07d7f0651bbbd9c2dfd44f33e44e0caea7a4e02b054f",
    4: "65673c9a8cf6b58445a9700993e9e9a00f2cf11fdb427d94ddf53d4433580227",
    5: "d48821f73b0902e55ab250eafc79b375f96f3f265342dd2024c6156c09bb1218",
}
ROUTES = ["八波段森林", "PCA 森林", "自适应布底", "参考工作台"]


class Metrics(C.Structure):
    _fields_ = [("count", C.c_uint64), ("decode_ms", C.c_double), ("ms", C.c_double)]


def read_planes(paths, average):
    images = []
    for path in paths:
        with Image.open(path) as image:
            if average and image.mode == "RGB":
                rgb = np.asarray(image)
                images.append(((rgb.sum(2, dtype=np.uint16) + 1) // 3).astype("uint8"))
            else:
                images.append(np.array(image.convert("L")))
    h, w = images[0].shape
    planes = (Plane * 8)(*[Plane(a.ctypes.data, a.size, w, h, a.strides[0], i)
                           for i, a in enumerate(images)])
    return images, planes


def reference(paths, shape, lib):
    h, w = shape
    mask = np.empty(shape, np.uint8)
    score = np.empty_like(mask)
    metrics = Metrics()
    error = C.create_unicode_buffer(512)
    names = (C.c_wchar_p * 8)(*map(str, paths))
    if lib.spectral_segment_v1(names, 8, w, h, 128, score.ctypes.data, mask.ctypes.data,
                               mask.size, C.byref(metrics), error, len(error)):
        raise ValueError(error.value)
    return mask, metrics.ms


def adaptive_mask(planes, shape):
    mask = np.empty(shape, np.uint8)
    score = np.empty_like(mask)
    result = Result()
    check(dll.ms_run_fabric(planes, 8, 128, mask.ctypes.data, score.ctypes.data, mask.size,
                            C.byref(result), None, None))
    return mask, result.ms


def check_historical(config):
    # 按 W 帧指纹找回旧编号，用原有标签和空台面选区评价自适应布底。
    # 同时核对森林的最终掩膜和原始得分，防止新路线意外改变旧设备的生产结果。
    with (OUT / "旧设备自适应复测.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["当前编号", "历史编号", "标注来源", "局部IoU", "局部漏提率", "局部误提率",
                         "台面误提像素", "台面选区像素"])
        for g in GROUPS:
            images, planes = group(g)
            shape = images[0].shape
            for route in [8, config["pca"]]:
                mask, votes, _ = predict(ROOT / f"models/rf{route}.yml", planes, shape,
                                         config["thresholds"][str(route)])
                archive = ROOT / f"results/无膜片/rf{route}/{g}"
                for name, actual in [("mask", mask), ("votes", votes)]:
                    with Image.open(archive / (name + ".png")) as saved:
                        np.testing.assert_array_equal(actual, np.asarray(saved),
                                                      err_msg=f"历史 {g} 号 rf{route}/{name}")
            mask, _ = adaptive_mask(planes, shape)
            truth, kind = labels(g, shape)
            region = mask[crop_region(shape, TABLE[g])] if g in TABLE else None
            writer.writerow([sample_folder(g).name, g, kind, *measure(mask, truth),
                             np.count_nonzero(region) if region is not None else "",
                             region.size if region is not None else ""])


def compare():
    config = json.loads((ROOT / "models/settings.json").read_text("utf8"))
    points = runpy.run_path(str(WORKBENCH / "tools/verify_samples.py"))["LANDMARKS"]
    lib = C.CDLL(str(WORKBENCH / "SpectralNative/bin/x64/Release/SpectralNative.dll"))
    lib.spectral_segment_v1.argtypes = [C.POINTER(C.c_wchar_p), C.c_int, C.c_int, C.c_int,
                                        C.c_int, ptr, ptr, C.c_uint64, C.POINTER(Metrics),
                                        C.POINTER(C.c_wchar), C.c_uint32]
    dll.ms_run_fabric.argtypes = [C.POINTER(Plane), C.c_int, C.c_uint8, ptr, ptr,
                                  C.c_int64, C.POINTER(Result), ptr, ptr]
    sheet = Image.new("RGB", (110 + 5 * 245, 36 + 5 * 164), "#eef1f3")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    for j, label in enumerate(["白光"] + ROUTES):
        draw.text((8, 104 + j * 164), label, font=font, fill="#203341")

    rows = []
    for g in range(1, 6):
        paths = [DATA / str(g) / (b + ".bmp") for b in BANDS]
        with paths[-1].open("rb") as white:
            if hashlib.file_digest(white, "sha256").hexdigest() != WHITE_SHA256[g]:
                raise ValueError(f"样品 {g} 的 W 图与复核点来源不同，请核对目录和样品身份")
        images, planes = read_planes(paths, False)
        h, w = images[0].shape
        x0 = 110 + (g - 1) * 245
        draw.text((x0 + 90, 5), f"样品 {g}", font=font, fill="#203341")
        sheet.paste(Image.fromarray(images[-1]).resize((240, 160)), (x0, 36))
        results = []
        for route in [8, config["pca"]]:
            mask, _, r = predict(ROOT / f"models/rf{route}.yml", planes, (h, w),
                                 config["thresholds"][str(route)])
            results.append((mask, r.ms))
        del images, planes
        images, planes = read_planes(paths, True)
        results.append(adaptive_mask(planes, (h, w)))
        del images, planes
        ref, ms = reference(paths, (h, w), lib)
        results.append((ref, ms))
        # 复用工作台按原始波段预选的点，1 号另含四个字孔及邻近笔画的像素。
        checks = [(int((x + .5) * w / 600), int((y + .5) * h / 400), v)
                  for coords, v in zip(points[g], (255, 0)) for x, y in coords]
        if g == 1:
            checks += [(3747, 1453, 255), (3754, 1496, 255), (3747, 1462, 0), (3758, 1512, 0)]
        for j, (name, (mask, ms)) in enumerate(zip(ROUTES, results)):
            cloth = mask[int(.1 * h):int(.9 * h), int(.9 * w):int(.98 * w)]
            iou = np.count_nonzero(mask & ref) / max(1, np.count_nonzero(mask | ref))
            rows.append([g, name, sum(mask[y, x] == v for x, y, v in checks), len(checks),
                         np.count_nonzero(cloth), round(iou, 4), round(ms, 1)])
            sheet.paste(Image.fromarray(mask).resize((240, 160)), (x0, 36 + (j + 1) * 164))
        print(f"完成样品 {g}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT / "五组对照.jpg", quality=92)
    with (OUT / "实拍对照.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["编号", "路线", "符合复核点", "复核点总数", "裸布误提像素", "与工作台掩膜IoU", "分割毫秒"])
        writer.writerows(rows)
    check_historical(config)


if __name__ == "__main__":
    compare()
