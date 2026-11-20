"""在原始分辨率独立校订参考标签。"""

import argparse
import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from PIL import Image, ImageTk, ImageDraw
from run import DATA, GROUPS, ROOT, BANDS, labels


class Editor:
    def __init__(self, win, group):
        self.win = win
        self.g = group
        self.band = tk.StringVar(value="W")
        self.paint = tk.StringVar(value="图文")
        self.radius = tk.IntVar(value=5)
        self.colors = {"图文": 1, "背景": 0, "无效台面": 2, "待确认": 255}
        self.path = ROOT / "references" / f"{group}.png"
        shape = (2048, 2448)
        arr, _ = labels(group, shape)
        self.lab = Image.fromarray(arr)
        self.zoom = 0.38
        self.x = 0
        self.y = 0
        self.polygon = []
        self.undo = None
        self.last = None
        self.dirty = False
        bar = ttk.Frame(win)
        bar.pack(fill="x")
        ttk.Label(bar, text=f"参考标注 · {group}").pack(side="left")
        cb = ttk.Combobox(bar, textvariable=self.band, values=BANDS, state="readonly", width=4)
        cb.pack(side="left")
        cb.bind("<<ComboboxSelected>>", lambda _: self.load())
        for name in self.colors:
            ttk.Radiobutton(bar, text=name, value=name, variable=self.paint).pack(side="left")
        ttk.Label(bar, text="笔半径(px)").pack(side="left")
        ttk.Spinbox(bar, from_=1, to=150, textvariable=self.radius, width=4).pack(side="left")
        ttk.Button(bar, text="载入候选", command=self.proposal).pack(side="left")
        ttk.Button(bar, text="撤销", command=self.revert).pack(side="left")
        ttk.Button(bar, text="保存参考", command=self.save).pack(side="left")
        self.overlay = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="叠加", variable=self.overlay, command=self.draw).pack(
            side="left"
        )
        self.canvas = tk.Canvas(win, bg="#242c30", width=1050, height=650)
        self.canvas.pack(fill="both", expand=True)
        ttk.Label(
            win,
            text="左键画笔 · 右键依次添加多边形顶点，Enter 填充，Esc 取消 · 滚轮缩放 · 中键拖动 · 绿=图文，蓝=背景，红=台面，无色=待确认",
        ).pack()
        c = self.canvas
        c.bind("<Button-1>", self.begin)
        c.bind("<B1-Motion>", self.stroke)
        c.bind("<ButtonRelease-1>", lambda _: setattr(self, "last", None))
        c.bind("<Button-3>", self.vertex)
        win.bind("<Return>", self.fill)
        win.bind("<Escape>", self.clear_poly)
        c.bind("<MouseWheel>", self.scale)
        c.bind("<Button-2>", self.pan_start)
        c.bind("<B2-Motion>", self.pan)
        c.bind("<Configure>", lambda _: self.draw())
        win.protocol("WM_DELETE_WINDOW", self.close)
        self.load()

    def load(self):
        with Image.open(DATA / str(self.g) / (self.band.get() + ".bmp")) as im:
            self.img = im.convert("L")
        self.draw()

    def point(self, e):
        return (int(self.x + e.x / self.zoom), int(self.y + e.y / self.zoom))

    def scale(self, e):
        px, py = self.point(e)
        self.zoom = max(0.12, min(10, self.zoom * (1.35 if e.delta > 0 else 1 / 1.35)))
        self.x = px - e.x / self.zoom
        self.y = py - e.y / self.zoom
        self.draw()

    def pan_start(self, e):
        self.anchor = (e.x, e.y, self.x, self.y)

    def pan(self, e):
        a, b, x, y = self.anchor
        self.x = x + (a - e.x) / self.zoom
        self.y = y + (b - e.y) / self.zoom
        self.draw()

    def draw(self):
        if not hasattr(self, "img"):
            return
        w = max(2, self.canvas.winfo_width())
        h = max(2, self.canvas.winfo_height())
        self.x = max(0, min(self.x, self.img.width - w / self.zoom))
        self.y = max(0, min(self.y, self.img.height - h / self.zoom))
        # 放大时只对可见视口采样，避免创建整幅放大图。
        box = (self.x, self.y, self.x + w / self.zoom, self.y + h / self.zoom)
        base = self.img.transform((w, h), Image.Transform.EXTENT, box).convert("RGB")
        if self.overlay.get():
            mask = np.array(self.lab.transform((w, h), Image.Transform.EXTENT, box, fillcolor=255))
            pix = np.array(base)
            for v, color in [(0, (40, 100, 220)), (1, (20, 225, 100)), (2, (245, 52, 56))]:
                use = mask == v
                pix[use] = pix[use] * 0.63 + np.array(color) * 0.37
            base = Image.fromarray(pix)
        self.photo = ImageTk.PhotoImage(base)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        pts = [((x - self.x) * self.zoom, (y - self.y) * self.zoom) for x, y in self.polygon]
        for a, b in pts:
            self.canvas.create_oval(a - 2, b - 2, a + 2, b + 2, fill="yellow")
        if len(pts) > 1:
            self.canvas.create_line(*pts, fill="yellow", width=2)

    def begin(self, e):
        self.undo = self.lab.copy()
        self.last = None
        self.stroke(e)

    def stroke(self, e):
        p = self.point(e)
        r = max(1, min(150, self.radius.get()))
        d = ImageDraw.Draw(self.lab)
        v = self.colors[self.paint.get()]
        if self.last:
            d.line([self.last, p], fill=v, width=r * 2 + 1)
        x, y = p
        d.ellipse((x - r, y - r, x + r, y + r), fill=v)
        self.last = p
        self.dirty = True
        self.draw()

    def vertex(self, e):
        self.polygon.append(self.point(e))
        self.draw()

    def fill(self, _):
        if len(self.polygon) >= 3:
            self.undo = self.lab.copy()
            ImageDraw.Draw(self.lab).polygon(self.polygon, fill=self.colors[self.paint.get()])
            self.dirty = True
        self.clear_poly(None)

    def clear_poly(self, _):
        self.polygon = []
        self.draw()

    def proposal(self):
        path = ROOT / "annotation-drafts" / str(self.g) / "proposal.png"
        if not path.exists():
            messagebox.showinfo("候选", "请先生成候选标注")
            return
        if not messagebox.askyesno(
            "载入候选", "将算法辅助候选载入画布？请逐区复核并将无法确认的像素标为待确认后再保存。"
        ):
            return
        self.undo = self.lab.copy()
        with Image.open(path) as im:
            self.lab = im.copy()
        self.dirty = True
        self.draw()

    def revert(self):
        if self.undo:
            self.lab, self.undo = self.undo, None
            self.dirty = True
            self.draw()

    def save(self):
        self.path.parent.mkdir(exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        self.lab.save(tmp, format="PNG")
        os.replace(tmp, self.path)
        self.dirty = False
        self.win.title(f"已保存 {self.path.name}")

    def close(self):
        if self.dirty:
            answer = messagebox.askyesnocancel("标注", "保存当前参考图？")
            if answer is None:
                return
            if answer:
                self.save()
        self.win.destroy()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("group", type=int, choices=GROUPS)
    args = p.parse_args()
    win = tk.Tk()
    win.title("多波段参考标注")
    Editor(win, args.group)
    win.mainloop()
