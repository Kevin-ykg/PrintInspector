"""在白光原图坐标下复测浅色字母和空白布带。"""

import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from run import DATA, OUT


def pixels(path):
    with Image.open(path) as image:
        return np.array(image)


def stroke_reference(white):
    # 该选区的紫外与白光比值能分开浅色笔画和织纹。
    # 比较两版在同一白光选区内的覆盖和溢出。
    uv = pixels(DATA / "12" / "UV.bmp")
    roi = np.s_[775:1010, 1060:1330]
    ratio = np.clip(255 * (uv[roi].astype(float) + 8) / (white[roi].astype(float) + 8), 0, 255)
    _, ref = cv2.threshold(
        cv2.medianBlur(ratio.astype(np.uint8), 5), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    return roi, ref != 0


def compare():
    baseline = OUT / "previous-masks"
    rows = []
    sheet = Image.new("RGB", (4 * 280, 4 * 215), "#192b34")
    draw = ImageDraw.Draw(sheet)
    for index, group in enumerate((12, 4, 5, 7)):
        white = pixels(DATA / str(group) / "W.bmp")
        if white.shape != (2048, 2448):
            raise ValueError(f"Fixed camera windows require 2448 x 2048 images (group {group})")
        for route in (8, 4):
            rel = Path(f"rf{route}") / str(group) / "mask.png"
            pair = [pixels(directory / rel) for directory in (baseline, OUT)]
            if group == 12:
                roi, ref = stroke_reference(white)
                for name, selection in (("stroke_proxy", ref), ("adjacent_background_proxy", ~ref)):
                    rows.append(
                        [
                            group,
                            route,
                            name,
                            int(selection.sum()),
                            *(int(np.count_nonzero(mask[roi][selection])) for mask in pair),
                        ]
                    )
                box = (1060, 775, 1330, 1010)
            else:
                # 计数前已在白光图中核对全高布带。4、7 号包含布面到台面的过渡，
                # 5 号右侧为深色布面。
                margin = np.zeros(white.shape, bool)
                margin[:, 2160:] = True
                if group in (4, 7):
                    margin[:, :210] = True
                rows.append(
                    [
                        group,
                        route,
                        "blank_margin",
                        int(margin.sum()),
                        *(int(np.count_nonzero(mask[margin])) for mask in pair),
                    ]
                )
                if route == 4:
                    # 局部裁剪仅用于展示；统计覆盖固定布带内的全部像素。
                    tiles = [
                        (x, y)
                        for x in ((0, 2240) if group != 5 else (2240,))
                        for y in range(0, 2048 - 255, 256)
                    ]
                    x, y = max(
                        tiles,
                        key=lambda p: np.count_nonzero(
                            pair[0][p[1] : p[1] + 256, p[0] : p[0] + 208]
                        ),
                    )
                    box = (x, y, x + 208, y + 256)
            if route == (8 if group == 12 else 4):
                context = Image.fromarray(white).convert("RGB")
                ImageDraw.Draw(context).rectangle(box, outline="orange", width=14)
                crops = [
                    context,
                    ImageOps.autocontrast(Image.fromarray(white).crop(box), cutoff=1),
                    *(Image.fromarray(mask).crop(box) for mask in pair),
                ]
                for column, (image, label) in enumerate(
                    zip(crops, ("W / crop", "W / local contrast", "Previous mask", "Current mask"))
                ):
                    image.thumbnail((274, 185), Image.Resampling.BOX)
                    left, top = column * 280, index * 215
                    sheet.paste(image, (left + (280 - image.width) // 2, top + 23))
                    draw.text((left + 5, top + 4), f"{group} | RF{route} | {label}", fill="white")
    with (OUT / "fabric-cases.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "group",
                "route",
                "region",
                "region_px",
                "previous_foreground_px",
                "current_foreground_px",
            )
        )
        writer.writerows(rows)
    sheet.save(OUT / "fabric-cases.png")


if __name__ == "__main__":
    compare()
