"""由人工种子为每件衣物独立生成可编辑的标签草稿。
校订并另存为参考标签后，才参与训练和评估。"""

import json
import cv2
import numpy as np
from PIL import Image
from run import GROUPS, ROOT, SEEDS, group, labels


def make(g):
    imgs, _ = group(g)
    h, w = imgs[-1].shape
    lab, _ = labels(g, (h, w))
    out = ROOT / "annotation-drafts" / str(g)
    out.mkdir(exist_ok=True, parents=True)
    Image.fromarray(lab).save(out / "seeds.png")
    # 按每组人工指出的墨区和背景区拟合光谱分布。
    # 类内混合分布覆盖不同墨色和布面不均匀性。
    samples = np.stack([a[::4, ::4] for a in imgs], axis=-1).astype(np.float32)
    low = lab[::4, ::4]
    means = []
    for cls in (0, 1):
        points = samples[low == cls]
        cv2.setRNGSeed(1789)
        k = min(6, len(points) // 30 + 1)
        _, idx, _ = cv2.kmeans(
            points, k, None, (cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001), 1, cv2.KMEANS_PP_CENTERS
        )
        components = []
        for c in range(k):
            z = points[idx[:, 0] == c]
            if len(z) < 9:
                continue
            cov = np.cov(z.T) + np.eye(8) * 6
            components.append(
                (z.mean(axis=0), np.linalg.inv(cov).astype("float32"), np.linalg.slogdet(cov)[1])
            )
        means.append(components)
    draft = np.full_like(lab, 255)
    for top in range(0, h, 32):
        end = min(top + 32, h)
        data = np.stack([a[top:end] for a in imgs], axis=-1).astype("float32")
        distances = []
        for components in means:
            costs = []
            for mu, inv, ld in components:
                d = data - mu
                costs.append(np.einsum("...i,ij,...j->...", d, inv, d, optimize=True) + ld)
            distances.append(np.min(costs, axis=0))
        delta = distances[0] - distances[1]
        dest = draft[top:end]
        dest[delta > 4] = 1
        dest[delta < -4] = 0
    # 基于白光及主成分总览的复核记录，只影响草稿。
    if g == 12:
        # 台面暗区连通到画面边界，图案内部深色文字另行识别。
        _, comp = cv2.connectedComponents((imgs[7] < 47).astype(np.uint8))
        edge_ids = np.unique(np.concatenate((comp[0], comp[-1], comp[:, 0], comp[:, -1])))
        table = np.isin(comp, edge_ids[edge_ids != 0])
        draft[table] = 2
        band = cv2.morphologyEx(
            table.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)
        )
        draft[band != 0] = 255
    if g == 5:
        # 黑布上浅色文字的亮度会随视场位置变化。
        smooth = cv2.GaussianBlur(imgs[7], (3, 3), 0)
        _, text = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        draft[:] = text // 255
        uncertain = cv2.morphologyEx(text, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
        draft[uncertain != 0] = 255
    if g in (3, 7, 9, 10, 15, 21, 22):
        # 已复核的图案范围之外，仅种子和复核窗口有已知归属。
        # 裁掉图案范围外的候选并留待复核。
        extents = {
            3: (53, 12, 257, 245),
            7: (36, 17, 270, 244),
            9: (4, 9, 296, 246),
            10: (55, 50, 264, 221),
            15: (42, 12, 278, 198),
            21: (1, 18, 319, 242),
            22: (29, 7, 290, 268),
        }
        x0, y0, x1, y1 = extents[g]
        draft[: round(y0 * h / 268)] = 255
        draft[round(y1 * h / 268) :] = 255
        draft[:, : round(x0 * w / 320)] = 255
        draft[:, round(x1 * w / 320) :] = 255
    if g in (3, 10, 22):
        regions = json.loads((ROOT / "evaluation/film-probes.json").read_text("utf8"))["groups"][
            str(g)
        ]
        for x0, y0, x1, y1 in regions:
            draft[
                round(y0 * h / 268) : round(y1 * h / 268), round(x0 * w / 320) : round(x1 * w / 320)
            ] = 0
    # 保留人工种子的判定，未观察的类别仍需复核。
    draft[lab != 255] = lab[lab != 255]
    Image.fromarray(draft).save(out / "proposal.png")


if __name__ == "__main__":
    for g in sorted(set(map(int, SEEDS)).intersection(GROUPS)):
        make(g)
        print("draft", g, flush=True)
