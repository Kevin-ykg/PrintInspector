"""从工程根目录运行。特征采样和推理均使用生产 DLL。"""

import argparse
import ctypes as C
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
BANDS = "UV B G AM R FR IR W".split()
SETS = json.loads((ROOT / "evaluation/sets.json").read_text("utf8"))
SEEDS = json.loads((ROOT / "evaluation/seed_regions.json").read_text("utf8"))["groups"]
DATA = Path(os.environ.get("PRINTVISION_DATA", ROOT.parent / "锐视光电" / "没有膜片"))
OUT = Path(os.environ.get("PRINTVISION_RESULTS", ROOT / "results" / "无膜片"))
MODELS = Path(os.environ.get("PRINTVISION_MODELS", ROOT / "models"))
RELATIVE = int(os.environ.get("PRINTVISION_RELATIVE", "1"))


# 历史参考图与种子使用原编号。生产图库可以任意改名，离线复测用 W 帧指纹恢复样品身份。
def sample_folders():
    folders = {int(d.name): d for d in DATA.iterdir() if d.is_dir() and d.name.isdecimal()}
    if DATA.resolve() != (ROOT.parent / "锐视光电" / "没有膜片").resolve():
        return folders
    fingerprints = json.loads((ROOT / "evaluation/historical_white_sha256.json").read_text("utf8"))
    known = {digest: int(key) for key, digest in fingerprints.items()}
    result = {}
    for folder in folders.values():
        white = folder / "W.bmp"
        if not white.is_file():
            continue
        with white.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in known:
            key = known[digest]
            if key in result:
                raise ValueError(f"历史样品 {key} 重复出现：{folder}")
            result[key] = folder
    return result


SAMPLE_FOLDERS = sample_folders()
GROUPS = sorted(SAMPLE_FOLDERS)


def sample_folder(g):
    return SAMPLE_FOLDERS[g]


class Plane(C.Structure):
    _fields_ = [("data", C.c_void_p), ("size", C.c_int64)] + [
        (k, C.c_int) for k in ["w", "h", "stride", "band"]
    ]


class Result(C.Structure):
    _fields_ = [
        ("ms", C.c_double),
        ("flags", C.c_int),
        ("count", C.c_int),
        ("shift", C.c_float),
        ("response", C.c_float),
    ]


libdir = Path(os.environ.get("PRINTVISION_NATIVE_DIR", ROOT / "build/vs/Release/native")).resolve()
search = os.add_dll_directory(str(libdir))
dll = C.CDLL(str(libdir / "PrintVision.dll"))
ptr = C.c_void_p
dll.ms_error.restype = C.c_char_p
dll.ms_load.argtypes = [C.c_wchar_p, C.POINTER(ptr)]
dll.ms_free.argtypes = [ptr]
dll.ms_train.argtypes = [ptr, ptr, C.c_int, C.c_int, C.c_wchar_p]
dll.ms_sample.argtypes = [C.POINTER(Plane), C.c_int, ptr, C.c_int, ptr]
dll.ms_sample_mode.argtypes = dll.ms_sample.argtypes + [C.c_int]
dll.ms_score_rows.argtypes = [ptr, ptr, C.c_int, ptr]
dll.ms_run.argtypes = [
    ptr,
    C.POINTER(Plane),
    C.c_int,
    C.c_uint8,
    ptr,
    ptr,
    C.c_int64,
    C.POINTER(Result),
    ptr,
]

# 允许选择历史 DLL 复算默认掩膜；只有自定义参数需要新版导出符号。
if hasattr(dll, "ms_run_configured"):
    dll.ms_run_configured.argtypes = dll.ms_run.argtypes + [C.c_int, C.c_int, C.c_int]


def check(code):
    if code:
        raise ValueError(dll.ms_error().decode(errors="replace"))


def split(g):
    family = next(k for k, ids in SETS["families"].items() if g in ids)
    return next(k for k in ("train", "development", "holdout") if family in SETS[k])


def group(g):
    imgs = []
    for b in BANDS:
        with Image.open(sample_folder(g) / (b + ".bmp")) as img:
            if img.mode not in ("L", "P", "RGB"):
                raise ValueError(f"{b}.bmp 的模式 {img.mode} 不受支持，需要灰度、索引色或 RGB 图像")
            imgs.append(np.array(img.convert("L")))
    planes = (Plane * 8)(
        *[
            Plane(a.ctypes.data, a.nbytes, a.shape[1], a.shape[0], a.strides[0], b)
            for b, a in enumerate(imgs)
        ]
    )
    return imgs, planes


def labels(g, shape):
    ref = ROOT / "references" / (str(g) + ".png")
    if ref.exists():
        with Image.open(ref) as im:
            lab = np.array(im)
        if lab.shape != shape or not np.isin(lab, [0, 1, 2, 255]).all():
            raise ValueError(f"Invalid reference {ref}")
        return lab, "reference"
    lab = np.full(shape, 255, np.uint8)
    if str(g) not in SEEDS:
        return lab, "unlabelled"
    h, w = shape
    for cls, key in enumerate(("background", "ink")):
        for x, y, r in SEEDS[str(g)][key] + (SEEDS[str(g)].get("film", []) if cls == 0 else []):
            cx, cy = round(x * w / 320), round(y * h / 268)
            rad = max(1, round(r * w / 320))
            yy, xx = np.ogrid[-rad : rad + 1, -rad : rad + 1]
            lab[cy - rad : cy + rad + 1, cx - rad : cx + rad + 1][
                xx * xx + yy * yy <= rad * rad
            ] = cls
    return lab, "visual_seed"


def predict(path, planes, shape, thr, settings=None):
    ctx = ptr()
    check(dll.ms_load(str(path), C.byref(ctx)))
    m = np.empty(shape, np.uint8)
    s = np.empty_like(m)
    r = Result()
    try:
        args = (ctx, planes, 8, thr, m.ctypes.data, s.ctypes.data, m.size, C.byref(r), None)
        if settings is None:
            check(dll.ms_run(*args))
        elif not hasattr(dll, "ms_run_configured"):
            raise ValueError("当前 DLL 不支持可调掩膜参数，请重新构建 Release 工程")
        else:
            check(dll.ms_run_configured(*args, *settings))
    finally:
        dll.ms_free(ctx)
    return m, s, r


def metrics(hist, t):
    tn, fn = hist[:, :t].sum(axis=1)
    fp, tp = hist[:, t:].sum(axis=1)
    return [tp / max(tp + fn + fp, 1), fn / max(fn + tp, 1), fp / max(fp + tn, 1)]


def run():
    OUT.mkdir(exist_ok=True, parents=True)
    MODELS.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(1789)
    xs = []
    ys = []
    ids = [g for g in GROUPS if str(g) in SEEDS or (ROOT / "references" / f"{g}.png").exists()]
    if not all(any(split(g) == part for g in ids) for part in ("train", "development", "holdout")):
        raise ValueError(
            "Training requires labelled groups in separate train, development and holdout families"
        )
    for g in ids:
        if split(g) != "train":
            continue
        imgs, p = group(g)
        lab, _ = labels(g, imgs[0].shape)
        for cls in (0, 1):
            idx = np.flatnonzero(lab == cls)
            idx = rng.choice(idx, min(7000, len(idx)), replace=False)
            xy = np.stack(np.unravel_index(idx, lab.shape)[::-1], axis=1).astype("int32").copy()
            features = np.empty((len(xy), 10), "float32")
            check(dll.ms_sample_mode(p, 8, xy.ctypes.data, len(xy), features.ctypes.data, RELATIVE))
            xs.append(features)
            ys.append(np.full(len(xy), cls, "int32"))
    X = np.concatenate(xs)
    Y = np.concatenate(ys)
    dev = {}
    rows = []
    development = []
    for g in ids:
        if split(g) != "development":
            continue
        imgs, p = group(g)
        lab, kind = labels(g, imgs[0].shape)
        flat = np.flatnonzero(lab < 2)
        xy = np.stack(np.unravel_index(flat, lab.shape)[::-1], axis=1).astype("int32").copy()
        features = np.empty((len(xy), 10), "float32")
        check(dll.ms_sample_mode(p, 8, xy.ctypes.data, len(xy), features.ctypes.data, RELATIVE))
        development.append((g, features, lab.ravel()[flat], kind))
    for route in [8, 2, 3, 4]:
        path = MODELS / f"rf{route}.yml"
        check(dll.ms_train(X.ctypes.data, Y.ctypes.data, len(Y), route + 10 * RELATIVE, str(path)))
        hists = []
        ctx = ptr()
        check(dll.ms_load(str(path), C.byref(ctx)))
        try:
            for g, features, truth, kind in development:
                score = np.empty(len(truth), np.uint8)
                check(dll.ms_score_rows(ctx, features.ctypes.data, len(truth), score.ctypes.data))
                hist = np.stack([np.bincount(score[truth == i], minlength=256) for i in (0, 1)])
                hists.append((g, hist, kind, ""))
        finally:
            dll.ms_free(ctx)

        def quality(t):
            return np.mean([metrics(h, t)[0] for _, h, _, _ in hists])

        t = max(range(32, 225, 8), key=lambda n: (quality(n), -abs(n - 128)))
        # 先用逐像素树票选阈值，再以最终掩膜比较 PCA 维数。
        refined = []
        for g, _, _, kind in development:
            imgs, p = group(g)
            lab, _ = labels(g, imgs[0].shape)
            mask, _, r = predict(path, p, lab.shape, t)
            hist = np.stack([np.bincount(mask[lab == i], minlength=256) for i in (0, 1)])
            values = metrics(hist, t)
            refined.append(values[0])
            rows.append([g, "development", kind, route, t, *values, r.ms])
        dev[route] = (float(np.mean(refined)), t)
        print("dev", route, dev[route], flush=True)
    best = max([2, 3, 4], key=lambda k: dev[k][0])
    config = {
        "route": 8,
        "pca": best,
        "thresholds": {str(k): v[1] for k, v in dev.items()},
        "status": "engineering review",
        "relative": RELATIVE,
    }
    (MODELS / "settings.json").write_text(json.dumps(config, indent=2), "utf8")
    for g in ids:
        if split(g) != "holdout":
            continue
        imgs, p = group(g)
        lab, k = labels(g, imgs[0].shape)
        for route in [8, best]:
            t = dev[route][1]
            m, _, r = predict(MODELS / f"rf{route}.yml", p, lab.shape, t)
            hist = np.stack([np.bincount(m[lab == i], minlength=256) for i in (0, 1)])
            rows.append([g, split(g), k, route, t, *metrics(hist, t), r.ms])
    with (OUT / "training.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            "group split label_basis route threshold IoU miss_rate false_positive_rate algorithm_ms".split()
        )
        writer.writerows(rows)


def batch():
    cfg = json.loads((MODELS / "settings.json").read_text())
    route = cfg["route"]
    t = cfg["thresholds"][str(route)]
    rows = []
    for g in GROUPS:
        start = time.perf_counter()
        imgs, p = group(g)
        m, s, r = predict(MODELS / f"rf{route}.yml", p, imgs[0].shape, t)
        dest = OUT / str(g)
        dest.mkdir(exist_ok=True, parents=True)
        Image.fromarray(m).save(dest / "mask.png", compress_level=1)
        Image.fromarray(s).save(dest / "score.png", compress_level=1)
        overlay = Image.fromarray(imgs[7]).convert("RGB")
        im = Image.fromarray(m)
        edge = np.array(im.filter(ImageFilter.MaxFilter(3))) != np.array(
            im.filter(ImageFilter.MinFilter(3))
        )
        overlay.paste((0, 237, 166), mask=Image.fromarray(edge))
        overlay.save(dest / "overlay.jpg", quality=88)
        rows.append(
            [g, r.flags, r.count, r.shift, r.response, r.ms, (time.perf_counter() - start) * 1000]
        )
        print("batch", g, flush=True)
    with (OUT / "batch.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            "group flags foreground_px max_shift min_response algorithm_ms read_to_export_ms".split()
        )
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train", "batch", "all"])
    a = ap.parse_args()
    if a.action in ["train", "all"]:
        if "PRINTVISION_MODELS" not in os.environ:
            MODELS = ROOT / "experiments" / "无膜片" / "models"
        run()
    if a.action in ["batch", "all"]:
        batch()
