"""用「多光谱」目录的 1～4 组对比两套程序，并逐像素核对当前 WPF 的 PNG 导出。"""

import csv
import ctypes as C
import json
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from compare_hardware import Metrics, WORKBENCH, read_planes, reference
from run import ROOT, Plane, Result, check, dll, ptr

DATA = ROOT.parent / "多光谱"
OUT = ROOT / "results/工作台对照"
EXE = ROOT / "desktop/bin/x64/Release/net10.0-windows/PrintInspector.exe"
BANDS = ("405", "457", "527", "600", "660", "730", "860", "w")


def camera_paths(folder):
    paths = []
    for band in BANDS:
        files = list(folder.glob(f"*-{band}-*.bmp"))
        if len(files) != 1:
            raise ValueError(f"{folder} 中 {band} 波段有 {len(files)} 个文件")
        paths.append(files[0])
    return paths


def verify_export(folder, white_path):
    with Image.open(folder / "mask.png") as image:
        assert image.mode == "L"
        mask = np.array(image)
    with Image.open(folder / "white-cutout.png") as cutout, Image.open(white_path) as white:
        assert cutout.mode == "RGBA" and cutout.size == white.size == mask.shape[::-1]
        # 按条带核对所有像素，避免同时展开两张完整彩色图。
        for y in range(0, mask.shape[0], 128):
            bottom = min(y + 128, mask.shape[0])
            rect = (0, y, mask.shape[1], bottom)
            a = mask[y:bottom]
            assert np.isin(a, [0, 255]).all()
            rgba = np.asarray(cutout.crop(rect))
            np.testing.assert_array_equal(rgba[:, :, 3], a)
            rgb = np.asarray(white.crop(rect).convert("RGB"))
            np.testing.assert_array_equal(rgba[a == 255, :3], rgb[a == 255])
            assert not rgba[a == 0, :3].any()
    return mask


def compare():
    ref = C.CDLL(str(WORKBENCH / "SpectralNative/bin/x64/Release/SpectralNative.dll"))
    ref.spectral_segment_v1.argtypes = [
        C.POINTER(C.c_wchar_p), C.c_int, C.c_int, C.c_int, C.c_int, ptr, ptr, C.c_uint64,
        C.POINTER(Metrics), C.POINTER(C.c_wchar), C.c_uint32,
    ]
    dll.ms_run_fabric_configured.argtypes = [
        C.POINTER(Plane), C.c_int, C.c_uint8, ptr, ptr, C.c_int64, C.POINTER(Result),
        ptr, ptr, C.c_int, C.c_int,
    ]
    sheet = Image.new("RGB", (1200, 4 * 225), "#eef1f3")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    staging = ROOT / "build/verify-workbench"
    staging.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for g in range(1, 5):
        sample = DATA / str(g)
        paths = camera_paths(sample)
        folder = staging / str(g)
        subprocess.run([str(EXE), "--verify-fabric-export", str(sample), str(folder)], check=True)
        mask = verify_export(folder, paths[-1])
        info = json.loads((folder / "metrics.json").read_text("utf8"))
        workbench, ref_ms = reference(paths, mask.shape, ref)

        # 面积设为零仅关闭当前版本的小分量过滤，配准及台面处理照常执行。
        images, planes = read_planes(paths, True)
        no_noise = np.empty_like(mask)
        score = np.empty_like(mask)
        r = Result()
        check(dll.ms_run_fabric_configured(planes, 8, 128, no_noise.ctypes.data, score.ctypes.data,
                                           score.size, C.byref(r), None, None, 0, 32))
        assert np.all(mask <= no_noise)
        default_fg, reference_fg = mask > 0, workbench > 0
        only_current = default_fg & ~reference_fg
        only_reference = reference_fg & ~default_fg
        iou = np.count_nonzero(default_fg & reference_fg) / max(1, np.count_nonzero(default_fg | reference_fg))
        iou_no_noise = np.count_nonzero(no_noise & workbench) / max(1, np.count_nonzero(no_noise | workbench))
        row = [g, mask.shape[1], mask.shape[0], round(iou, 5), round(iou_no_noise, 5),
               np.count_nonzero(only_reference), np.count_nonzero(only_current),
               np.count_nonzero(no_noise != mask), round(info["nativeMs"], 1), round(ref_ms, 1),
               json.dumps(info["offsets"])]
        rows.append(row)
        print(row, flush=True)
        diff = np.zeros((*mask.shape, 3), np.uint8)
        diff[only_reference] = (236, 95, 67)
        diff[only_current] = (50, 206, 225)
        for i, (title, image) in enumerate([
            ("白光", Image.fromarray(images[-1])), ("工作台", Image.fromarray(workbench)),
            ("当前项目", Image.fromarray(mask)), ("红：工作台独有  青：当前独有", Image.fromarray(diff)),
        ]):
            image.thumbnail((286, 190))
            x, y = i * 300, (g - 1) * 225
            draw.text((x + 10, y + 5), f"{g} 号  {title}", font=font, fill="#172732")
            sheet.paste(image, (x + (300 - image.width) // 2, y + 29))

    sheet.save(OUT / "四组对照.jpg", quality=94)
    with (OUT / "四组对照.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["目录编号", "宽", "高", "默认掩膜IoU", "关闭孤点过滤IoU",
                         "工作台独有像素", "当前独有像素", "孤点过滤删除像素",
                         "当前DLL毫秒", "工作台分割毫秒", "当前配准位移"])
        writer.writerows(rows)


if __name__ == "__main__":
    compare()
