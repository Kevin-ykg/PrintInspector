"""固定白光选区内沿扫描线量笔画宽度与中心偏差。以两侧稳定的半亮度边界作光学参照。"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from run import DATA, OUT

# 原分辨率 W 坐标 (左,上,右,下)，末项指明预期墨层是否较背景亮。
REGIONS = {
    2: (500, 1238, 1735, 1555, True),
    5: (280, 100, 2100, 530, True),
    12: (680, 1250, 1240, 1415, False),
    15: (440, 775, 2050, 910, True),
}


def measure(white, mask, bright):
    gray = cv2.GaussianBlur(white, (5, 5), 1).astype(float)
    if not bright:
        gray = 255 - gray
    threshold, _ = cv2.threshold(gray.astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
    widths, centers, edge_errors = [], [], []
    eligible = 0
    for image, binary in ((gray, mask), (gray.T, mask.T)):
        for row, prediction in zip(image[::5], binary[::5]):
            edges = np.flatnonzero(np.diff((row > threshold).astype(int))) + 1
            jumps = np.diff((prediction != 0).astype(int))
            for ix, (left, right) in enumerate(zip(edges, edges[1:])):
                if row[left] <= threshold or right - left < 14 or left < 9 or right + 9 >= len(row):
                    continue
                if (ix and left - edges[ix - 1] < 8) or (
                    ix + 2 < len(edges) and edges[ix + 2] - right < 8
                ):
                    continue
                optical = []
                for x, sign in ((left, 1), (right, -1)):
                    a, b = row[x - 7 : x - 4], row[x + 4 : x + 7]
                    delta = b.mean() - a.mean()
                    if sign * delta < 35 or max(np.ptp(a), np.ptp(b)) > abs(delta) * 0.3:
                        break
                    mid = (a.mean() + b.mean()) / 2
                    seg = row[x - 4 : x + 5]
                    crossings = np.flatnonzero((seg[:-1] - mid) * sign <= 0)
                    valid = [i for i in crossings if (seg[i + 1] - mid) * sign > 0]
                    if len(valid) != 1:
                        break
                    i = valid[0]
                    optical.append(x - 4 + i + (mid - seg[i]) / (seg[i + 1] - seg[i]))
                if len(optical) != 2:
                    continue
                eligible += 1
                actual = []
                for x, sign, ref in zip((left, right), (1, -1), optical):
                    candidates = np.flatnonzero(jumps[x - 8 : x + 8] == sign) + x - 7.5
                    if len(candidates):
                        actual.append(candidates[np.argmin(abs(candidates - ref))])
                if len(actual) == 2:
                    a, b = np.array(actual) - optical
                    widths.append(b - a)
                    centers.append(abs((a + b) / 2))
                    edge_errors.extend((abs(a), abs(b)))
    return [
        eligible,
        len(widths),
        *[
            round(float(x), 3)
            for x in (
                np.median(widths),
                np.mean(np.abs(widths)),
                np.mean(centers),
                np.percentile(edge_errors, 95),
            )
        ],
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="?", default=OUT)
    parser.add_argument("--before", type=Path, help="同结构的旧结果目录，用于逐阶段比较")
    args = parser.parse_args()
    with (args.results / "edge-profiles.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "group",
                "route",
                "stage",
                "reference_profiles",
                "matched_profiles",
                "median_signed_width_px",
                "mean_abs_width_px",
                "mean_abs_center_px",
                "p95_abs_edge_px",
            ]
        )
        for group, (x, y, r, b, bright) in REGIONS.items():
            w = np.asarray(Image.open(DATA / str(group) / "W.bmp").convert("L"))[y:b, x:r]
            for route in (8, 4):
                folder = Path(f"rf{route}") / str(group)
                stages = [("votes", args.results / folder / "votes.png")]
                if args.before:
                    stages.append(("before", args.before / folder / "mask.png"))
                stages.append(("after", args.results / folder / "mask.png"))
                for stage, path in stages:
                    m = np.asarray(Image.open(path))[y:b, x:r]
                    if stage == "votes":
                        m = m >= (112 if route == 8 else 104)
                    row = [group, route, stage, *measure(w, m, bright)]
                    writer.writerow(row)
                    print(row)


if __name__ == "__main__":
    main()
