"""有膜片试验：复用生产 DLL，输出逐组掩膜、叠加图和已有选区的计数。"""

import csv
import json

import numpy as np
from PIL import Image, ImageDraw

from run import DATA, GROUPS, MODELS, OUT, ROOT, group, labels, predict


def review():
    if DATA.name != "有膜片" or OUT.resolve() == (ROOT / "results/无膜片").resolve():
        raise ValueError(
            "请将 PRINTVISION_DATA 指向有膜片目录，PRINTVISION_RESULTS 指向独立试验目录"
        )
    probes = json.loads((ROOT / "evaluation/film-probes.json").read_text("utf8"))["groups"]
    cfg = json.loads((MODELS / "settings.json").read_text("utf8"))
    OUT.mkdir(parents=True, exist_ok=True)
    pages = {}
    rows = []
    for i, g in enumerate(GROUPS):
        images, planes = group(g)
        h, w = images[-1].shape
        lab, basis = labels(g, (h, w))
        bare_film = np.zeros((h, w), bool)
        for x0, y0, x1, y1 in probes.get(str(g), []):
            bare_film[
                round(y0 * h / 268) : round(y1 * h / 268), round(x0 * w / 320) : round(x1 * w / 320)
            ] = True
        bare_film &= lab == 255
        for route in (8, cfg["pca"]):
            threshold = cfg["thresholds"][str(route)]
            # 裸膜与图文都可能有高票；先检验提高阈值能否减少裸膜误提且不损伤已标图文。
            for t in [threshold, min(threshold + 32, 255)] if str(g) in probes else [threshold]:
                mask, _, info = predict(MODELS / f"rf{route}.yml", planes, (h, w), t)
                rows.append(
                    [
                        g,
                        route,
                        t,
                        basis,
                        info.flags,
                        info.count,
                        int(bare_film.sum()),
                        np.count_nonzero(mask[bare_film]),
                        np.count_nonzero(lab == 1),
                        np.count_nonzero(mask[lab == 1]),
                        np.count_nonzero(lab == 0),
                        np.count_nonzero(mask[lab == 0]),
                    ]
                )
                if t != threshold:
                    continue
                dest = OUT / f"rf{route}" / str(g)
                dest.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(dest / "mask.png", compress_level=1)
                view = Image.fromarray(images[-1]).convert("RGB")
                view.paste((0, 220, 160), mask=Image.fromarray(mask // 3))
                view.save(dest / "overlay.jpg", quality=85)
                page = i // 10
                if page not in pages:
                    pages[page] = Image.new("RGB", (4 * 230, 5 * 210), "#192c34")
                col, row = (i % 2) * 2 + int(route != 8), (i % 10) // 2
                pages[page].paste(view.resize((230, 190)), (col * 230, row * 210 + 20))
                ImageDraw.Draw(pages[page]).text(
                    (col * 230 + 6, row * 210 + 3), f"{g}  RF{route}", fill="white"
                )
        print(g, flush=True)
    for idx, im in pages.items():
        im.save(OUT / f"overview-{idx + 1}.jpg", quality=90)
    with (OUT / "review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            "group route threshold label_basis flags foreground_px film_px film_fp ink_px ink_tp background_px background_fp".split()
        )
        writer.writerows(rows)


def compare_difficult():
    """在大片背景与浅色笔画上单独试参，结果不写入生产配置。"""
    cfg = json.loads((MODELS / "settings.json").read_text("utf8"))
    rows = []
    for g in (6, 23):
        images, planes = group(g)
        h, w = images[-1].shape
        region = np.zeros((h, w), bool)
        if g == 6:
            region[160:1830, :140] = True
        else:
            # 在白光原图中逐一核对的浅色笔画内部；每个中心取 9×9 像素。
            for x, y in ((750, 276), (867, 282), (1091, 246), (1474, 266)):
                region[y - 4 : y + 5, x - 4 : x + 5] = True
        for route in (8, cfg["pca"]):
            t = cfg["thresholds"][str(route)]
            default = np.asarray(Image.open(OUT / f"rf{route}" / str(g) / "mask.png"))
            settings = (512, 80, 46) if g == 6 else (96, 2048, 100)
            for tag, threshold, params in (
                ("default", t, (96, 80, 46)),
                ("cleanup", t, settings),
                ("threshold", t + (32 if g == 6 else -32), (96, 80, 46)),
            ):
                m = (
                    default
                    if tag == "default"
                    else predict(MODELS / f"rf{route}.yml", planes, (h, w), threshold, params)[0]
                )
                rows.append(
                    [
                        g,
                        route,
                        tag,
                        threshold,
                        *params,
                        int(region.sum()),
                        int(np.count_nonzero(m[region])),
                        int(np.count_nonzero(m)),
                    ]
                )
    with (OUT / "difficult.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            "group route trial threshold noise_area hole_area allowance roi_px roi_foreground total_foreground".split()
        )
        writer.writerows(rows)


if __name__ == "__main__":
    review()
    compare_difficult()
