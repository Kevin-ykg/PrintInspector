"""逐组运行两条生产路线并比较原始投票和最终掩膜。"""

import csv
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from run import GROUPS, OUT, MODELS, group, labels, predict, split

# 在 320×268 缩略图上确认的空台面内部矩形。
# 5 号近黑布料仍有明显红外响应，不能仅凭白光判为空台面。
TABLE = {
    2: (30, 1, 160, 5),
    4: (1, 30, 5, 200),
    7: (314, 30, 318, 215),
    8: (40, 2, 250, 9),
    12: (64, 251, 119, 263),
}


def small_components(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] <= 32))


def crop_region(shape, box):
    h, w = shape
    x0, y0, x1, y1 = box
    return np.s_[
        round(y0 * h / 268) : round(y1 * h / 268), round(x0 * w / 320) : round(x1 * w / 320)
    ]


def measure(mask, lab):
    pos = lab == 1
    neg = lab == 0
    tp = np.count_nonzero(mask[pos])
    fp = np.count_nonzero(mask[neg])
    fn = int(pos.sum()) - tp
    return (
        (tp / (tp + fp + fn) if tp + fp + fn else None),
        (fn / pos.sum() if pos.any() else None),
        (fp / neg.sum() if neg.any() else None),
    )


def run_review():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((MODELS / "settings.json").read_text())
    table = []
    pages = {
        r: Image.new("RGB", (3 * 320, ((len(GROUPS) + 2) // 3) * 200), "#193039")
        for r in (8, cfg["pca"])
    }
    for idx, g in enumerate(GROUPS):
        images, planes = group(g)
        h, w = images[7].shape
        lab, basis = labels(g, (h, w))
        for route, page in pages.items():
            threshold = cfg["thresholds"][str(route)]
            mask, score, result = predict(MODELS / f"rf{route}.yml", planes, (h, w), threshold)
            raw = (score >= threshold).astype("uint8") * 255
            for stage, m in (("votes", raw), ("final", mask)):
                region = m[crop_region(m.shape, TABLE[g])] if g in TABLE else None
                table.append(
                    [
                        g,
                        split(g),
                        basis,
                        route,
                        stage,
                        *measure(m, lab),
                        small_components(m),
                        np.count_nonzero(region) if region is not None else None,
                        region.size if region is not None else None,
                        result.ms if stage == "final" else None,
                    ]
                )
            dest = OUT / f"rf{route}" / str(g)
            dest.mkdir(parents=True, exist_ok=True)
            Image.fromarray(mask).save(dest / "mask.png", compress_level=1)
            Image.fromarray(score).save(dest / "votes.png", compress_level=1)
            im = Image.fromarray(images[7]).convert("RGB")
            im.paste((20, 205, 145), mask=Image.fromarray(mask // 6))
            contours = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            im.paste((20, 245, 155), mask=Image.fromarray(contours))
            im.save(dest / "overlay.jpg", quality=88)
            x = idx % 3 * 320
            y = idx // 3 * 200
            left = Image.fromarray(raw).resize((160, 166))
            right = Image.fromarray(mask).resize((160, 166))
            page.paste(left, (x, y + 25))
            page.paste(right, (x + 160, y + 25))
            ImageDraw.Draw(page).text(
                (x + 4, y + 4), f"{g}   |   raw votes         refined", fill="white"
            )
            print(f"{g} / {route}", flush=True)
    for route, page in pages.items():
        page.save(OUT / f"comparison-rf{route}.jpg", quality=92)
    with (OUT / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            "group split label_basis route stage IoU miss_rate false_positive_rate components_le32 table_false_pixels table_pixels ms".split()
        )
        writer.writerows(table)


if __name__ == "__main__":
    run_review()
