"""汇总逐组验证结果，包含高对比区域的边缘近似指标。"""

import csv
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from run import sample_folder, GROUPS, MODELS, OUT


def edge_score(mask, reference):
    def edges(image):
        border = image != cv2.erode(image, np.ones((3, 3), np.uint8))
        border[:3] = border[-3:] = False
        border[:, :3] = border[:, -3:] = False
        return border

    def covered(a, b):
        near = cv2.dilate(b.astype(np.uint8), np.ones((5, 5), np.uint8)) != 0
        return np.count_nonzero(a & near) / max(1, a.sum())

    a, b = edges(mask), edges(reference)
    precision, recall = covered(a, b), covered(b, a)
    return 2 * precision * recall / max(1e-12, precision + recall)


def report():
    with (OUT / "comparison.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    cfg = json.loads((MODELS / "settings.json").read_text())
    routes = (8, cfg["pca"])
    assert len(rows) == 4 * len(GROUPS)
    lines = [
        "# 无膜片样品验证",
        "",
        f"样品编号为 {', '.join(map(str, GROUPS))}。同一模型的原始投票按原阈值二值化，与最终掩膜在同一坐标比较。同一组固定投票经过底布估计与空间处理后得到最终掩膜。本次固定选区的新旧掩膜对照见 docs/样品复核.md，早期布边实验见《布边与孤立点复核》。",
        "",
        "| 模型及处理 | ≤32 像素前景连通域总数 | 黑边选区误提像素 / 选区面积 | 已标注组宏平均 IoU |",
        "|---|---:|---:|---:|",
    ]
    for route in routes:
        for stage in ("votes", "final"):
            sub = [r for r in rows if int(r["route"]) == route and r["stage"] == stage]
            areas = sum(int(r["components_le32"]) for r in sub)
            errors = sum(int(r["table_false_pixels"] or 0) for r in sub)
            total = sum(int(r["table_pixels"] or 0) for r in sub)
            iou = np.mean([float(r["IoU"]) for r in sub if r["IoU"]])
            name = "八波段" if route == 8 else f"PCA {route} 维"
            lines.append(
                f"| {name} · {'原始' if stage == 'votes' else '优化后'} | {areas:,} | {errors:,} / {total:,} | {iou:.2%} |"
            )
    lines += [
        "",
        "小连通域总数是碎片指标，其中可能有真实小点。黑边评价使用 2、4、7、8、12 号的人工定位空台面选区。IoU 只统计 2、5、7、12、15 号已有局部标注，含历史训练款、开发款、留出款；这些数据已用于研发观察。",
        "",
        "## 图案边缘与内部",
        "",
        "5 号上部高对比文字，取原图 x=[280,2100)、y=[100,530) 的白光 Otsu 轮廓作近似参考，计算横纵各容许 2 像素的边界 F1。该单项用于检查清晰笔画的边缘贴合度。",
    ]
    crop = np.s_[100:530, 280:2100]
    white = np.array(Image.open(sample_folder(5) / "W.bmp"))[crop]
    _, reference = cv2.threshold(white, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    for route in routes:
        path = OUT / f"rf{route}" / "5"
        threshold = cfg["thresholds"][str(route)]
        raw = (np.array(Image.open(path / "votes.png"))[crop] >= threshold).astype(np.uint8) * 255
        final = np.array(Image.open(path / "mask.png"))[crop]
        lines.append(
            f"- {'八波段' if route == 8 else 'PCA'}的边界 F1 从 {edge_score(raw, reference):.3f} 变为 {edge_score(final, reference):.3f}。"
        )
    lines += [
        "",
        "12 号标志内漏提的色块可对照 detail.jpg。原样设计内孔和可能的真实缺墨都需要保留，算法只做有限的、受光谱证据约束的填补。个别样品的局部 IoU 会下降，见逐组数值。",
        "",
        "## 逐组局部 IoU",
        "",
        "| 编号 | 历史划分 | 八波段原始 → 优化 | PCA 原始 → 优化 |",
        "|---|---|---:|---:|",
    ]
    for g in GROUPS:
        sub = [r for r in rows if int(r["group"]) == g and r["IoU"]]
        if not sub:
            continue
        vals = [f"{float(r['IoU']):.1%}" for r in sub]
        lines.append(f"| {g} | {sub[0]['split']} | {vals[0]} → {vals[1]} | {vals[2]} → {vals[3]} |")
    lines += [
        "",
        "## 推理耗时",
        "",
        "Release x64，按当前无膜片目录逐组运行，包含原生预处理、投票和掩膜优化。",
    ]
    for route in routes:
        times = [
            float(r["ms"]) / 1000
            for r in rows
            if r["stage"] == "final" and int(r["route"]) == route
        ]
        lines.append(
            f"- {'八波段' if route == 8 else 'PCA'} P50 {np.median(times):.2f} 秒，P95 {np.percentile(times, 95):.2f} 秒。"
        )
    benchmark = OUT / "benchmark.csv"
    if benchmark.exists():
        with benchmark.open(encoding="utf8", newline="") as stream:
            bench = list(csv.DictReader(stream))
        for route in routes:
            runs = [r for r in bench if int(r["route"]) == route]
            if len(runs) != len(GROUPS) * (2 if route == 8 else 1):
                continue
            peak = max(float(r["peak_private_mib"]) for r in runs)
            elapsed = [float(r["read_to_render_ms"]) / 1000 for r in runs]
            lines.append(
                f"- WPF {len(runs)} 次读盘到离屏渲染，{'八波段' if route == 8 else 'PCA'} P95 {np.percentile(elapsed, 95):.2f} 秒，采样峰值私有内存 {peak:.1f} MiB。"
            )
    (OUT / "验证报告.md").write_text("\n".join(lines) + "\n", "utf8")
    cases = [
        (5, (280, 100, 2100, 530)),
        (7, (470, 1510, 1810, 1810)),
        (11, (90, 175, 840, 900)),
        (12, (610, 790, 1710, 1460)),
    ]
    sheet = Image.new("RGB", (4 * 268, len(cases) * 173), "#17303b")
    draw = ImageDraw.Draw(sheet)
    for y, (g, box) in enumerate(cases):
        base = OUT / "rf8" / str(g)
        for x, (caption, path) in enumerate(
            [
                ("W", sample_folder(g) / "W.bmp"),
                ("Raw votes", base / "votes.png"),
                ("Mask", base / "mask.png"),
                ("Overlay", base / "overlay.jpg"),
            ]
        ):
            with Image.open(path) as im:
                part = im.crop(box)
            if x == 1:
                part = part.point(lambda v: 255 if v >= cfg["thresholds"]["8"] else 0)
            part.thumbnail((266, 149))
            sheet.paste(part, (x * 268, y * 173 + 22))
            draw.text((x * 268 + 4, y * 173 + 3), f"{g}  {caption}", fill="white")
    sheet.save(OUT / "detail.jpg", quality=93)


if __name__ == "__main__":
    report()
