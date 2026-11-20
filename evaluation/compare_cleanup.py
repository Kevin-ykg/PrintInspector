"""在人工确认的布面及布边选区比较两版掩膜。"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from run import DATA, GROUPS, MODELS, OUT
import json

# 坐标来自 320×268 预览图，布边选区横跨实际边界。
EDGE = {
    2: [(0, 0, 320, 25), (0, 160, 16, 268)],
    4: [(0, 0, 27, 268), (282, 0, 320, 268)],
    7: [(0, 0, 24, 268), (282, 0, 320, 268)],
    8: [(0, 0, 320, 37), (0, 236, 320, 268)],
    11: [(304, 0, 320, 268)],
    12: [(0, 0, 34, 268), (252, 0, 320, 268), (34, 206, 252, 268)],
}
CLOTH = {
    2: [(18, 38, 56, 172), (240, 42, 296, 204)],
    4: [(39, 88, 65, 173), (226, 112, 279, 170)],
    7: [(40, 112, 70, 189), (226, 105, 279, 171)],
    8: [(24, 59, 58, 200), (236, 55, 285, 197)],
    11: [(122, 170, 159, 201), (23, 87, 46, 124)],
    12: [(43, 111, 72, 180), (216, 84, 253, 164)],
    15: [(72, 202, 235, 245)],
}


def select(shape, boxes):
    region = np.zeros(shape, bool)
    height, width = shape
    for x0, y0, x1, y1 in boxes:
        region[
            round(y0 * height / 268) : round(y1 * height / 268),
            round(x0 * width / 320) : round(x1 * width / 320),
        ] = True
    return region


def compare(previous):
    cfg = json.loads((MODELS / "settings.json").read_text())
    rows = []
    routes = [8, cfg["pca"]]
    for route in routes:
        for g in GROUPS:
            for version, root in [("previous", previous), ("current", OUT)]:
                path = root / f"rf{route}" / str(g) / "mask.png"
                with Image.open(path) as im:
                    mask = np.asarray(im)
                edge = select(mask.shape, EDGE.get(g, []))
                cloth = select(mask.shape, CLOTH.get(g, []))
                _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                rows.append(
                    [
                        route,
                        g,
                        version,
                        int(edge.sum()),
                        int(np.count_nonzero(mask[edge])),
                        int(cloth.sum()),
                        int(np.count_nonzero(mask[cloth])),
                        int(np.count_nonzero(stats[1:, cv2.CC_STAT_AREA] <= 32)),
                    ]
                )
    with (OUT / "cleanup.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "route",
                "group",
                "version",
                "edge_area",
                "edge_false_px",
                "cloth_area",
                "cloth_false_px",
                "components_le32",
            ]
        )
        writer.writerows(rows)
    lines = [
        "# 布边与孤立点复核",
        "",
        "采用当前无膜片组 "
        + ", ".join(map(str, GROUPS))
        + " 的完整掩膜。上一版与本版使用相同模型和阈值。",
        "布边选区跨越布料与暗台面的交界。空布选区位于图文以外。两类区域由样品白光图定位，统计选区内被判为前景的像素。小连通域在全图计数，其中可能包含真实印点。",
        "",
        "| 路线 | 布边选区前景像素 | 空布选区前景像素 | 全图 ≤32 像素连通域 |",
        "|---|---:|---:|---:|",
    ]
    for route in routes:
        totals = [
            np.array([r[3:] for r in rows if r[0] == route and r[2] == v]).sum(0)
            for v in ("previous", "current")
        ]
        a, b = totals
        lines.append(
            f"| {'八波段' if route == 8 else 'PCA'} | {a[1]:,} → {b[1]:,} / {b[0]:,} | {a[3]:,} → {b[3]:,} / {b[2]:,} | {a[4]:,} → {b[4]:,} |"
        )
    lines += [
        "",
        "检测范围仅覆盖所列选区。真实孤立小印点与同尺寸、同光谱的布面噪点存在混淆风险，最小保留面积的工艺验收需要标定。",
        "回归样例另行检查紧邻台面的长细线、横穿边界方向的印刷笔画、文字内孔与独立 8×8 像素印点。",
    ]
    (OUT / "布边与孤立点复核.md").write_text("\n".join(lines) + "\n", "utf8")
    # 白光、旧掩膜、新掩膜使用相同裁剪窗口。
    cases = [
        (4, (0, 170, 20, 259)),
        (7, (0, 64, 21, 109)),
        (12, (80, 205, 151, 233)),
        (7, (76, 204, 228, 239)),
        (11, (22, 14, 115, 90)),
    ]
    sheet = Image.new("RGB", (3 * 300, 150 * len(cases)), "#1a2c32")
    draw = ImageDraw.Draw(sheet)
    for row, (g, rect) in enumerate(cases):
        for col, (path, caption) in enumerate(
            [
                (DATA / str(g) / "W.bmp", "W"),
                (previous / "rf8" / str(g) / "mask.png", "Previous"),
                (OUT / "rf8" / str(g) / "mask.png", "Current"),
            ]
        ):
            with Image.open(path) as im:
                area = tuple(
                    round(v * im.size[i % 2] / (320, 268)[i % 2]) for i, v in enumerate(rect)
                )
                crop = im.crop(area)
                crop.thumbnail((297, 126))
            x = col * 300
            y = row * 150
            draw.text((x + 6, y + 3), f"{g}  {caption}", fill="white")
            sheet.paste(crop, (x + 5, y + 20))
    sheet.save(OUT / "cleanup.jpg", quality=94)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "previous", type=Path, help="Directory containing previous rf8 and PCA mask folders"
    )
    compare(parser.parse_args().previous)
