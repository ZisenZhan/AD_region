"""Plot the frozen path honestly and compare development-selected panels with bases."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve

OUT = Path(__file__).resolve().parents[1]
R, FIG, QA = [OUT / n for n in ("结果", "图件", "核对")]
FIG.mkdir(exist_ok=True)
REGIONS = {"frontal_prefrontal": ("额叶前额叶", "Frontal / prefrontal"),
           "temporal_neocortex": ("颞叶新皮层", "Temporal neocortex"),
           "hippocampus": ("海马", "Hippocampus")}
DEV, EXT = "#3C7883", "#B2676C"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 7,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": .65, "legend.frameon": False})
checks = []


def read(p):
    return pd.read_csv(p, float_precision="round_trip")


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text():
            b = t.get_window_extent(renderer)
            if b.x0 < -.5 or b.y0 < -.5 or b.x1 > fig.bbox.width+.5 or b.y1 > fig.bbox.height+.5:
                outside.append(t.get_text())
    assert not outside, (name, outside)
    px = np.asarray(fig.canvas.buffer_rgba())
    for ext in ("png", "svg", "pdf"):
        fig.savefig(FIG / (name+"."+ext), dpi=400, facecolor="white")
    checks.append(dict(figure=name, text_outside_canvas=outside,
                       nonwhite_fraction=float((px[:, :, :3]<245).any(axis=2).mean()),
                       editable_svg="<text" in (FIG / (name+".svg")).read_text(encoding="utf-8")))
    plt.close(fig)


def path_plot(curve, summary):
    fig = plt.figure(figsize=(7.2, 8.3))
    gs = fig.add_gridspec(3, 1, left=.105, right=.98, bottom=.075, top=.91, hspace=.58)
    for i, (region, (_, en)) in enumerate(REGIONS.items()):
        z = curve.loc[curve.region.eq(region)].sort_values("k")
        row = summary.loc[summary.region.eq(region)].iloc[0]
        k = int(row.selected_k)
        inner = gs[i, 0].subgridspec(2, 1, height_ratios=[3, 1], hspace=.12)
        ax, bx = fig.add_subplot(inner[0, 0]), fig.add_subplot(inner[1, 0])
        for column, color in [("dev_macro", DEV), ("external_auc", EXT)]:
            solid, tail = z.loc[z.k.le(k)], z.loc[z.k.ge(k)]
            ax.plot(solid.k, solid[column], color=color, lw=1.2)
            ax.plot(tail.k, tail[column], color=color, lw=1.05, ls="--")
            base = z.loc[z.k.eq(6)].iloc[0]
            selected = z.loc[z.k.eq(k)].iloc[0]
            ax.plot(6, base[column], 'o', mfc="white", mec=color, ms=4, mew=.9)
            ax.plot(k, selected[column], 'D', color=color, ms=4)
        ax.axvline(k, color="#A6AFB3", lw=.65, ls=":")
        ax.set_xlim(5, 61)
        ax.set_ylim(.5, 1.025)
        ax.set_yticks([.5, .75, 1.0])
        ax.set_xticks([6, 10, 20, 30, 40, 50, 60], [])
        ax.set_ylabel("AUC")
        ax.set_title(en, fontsize=8.5, loc="left", pad=6)
        ax.text(-.085, 1.03, chr(97+i*2), transform=ax.transAxes, weight="bold", fontsize=10)
        ax.grid(axis="y", color="#E9EDEF", lw=.4)
        increments = z.loc[z.k.gt(6)]
        bx.bar(increments.k, increments.dev_delta, width=.72,
               color=np.where(increments.dev_delta.ge(0), "#A4C5CC", "#D5A9AF"), linewidth=0)
        bx.axhline(0, lw=.65, color="#626E74")
        bx.axvline(k, color="#A6AFB3", lw=.65, ls=":")
        bx.set_xlim(5, 61)
        bx.set_xticks([6, 10, 20, 30, 40, 50, 60])
        bx.set_ylabel("Dev ΔAUC", fontsize=6.5)
        bx.set_xlabel("Number of genes" if i == 2 else "")
        bx.tick_params(labelsize=6, length=3)
        ax.tick_params(labelsize=6, length=3)
        bx.text(-.085, 1.01, chr(98+i*2), transform=bx.transAxes, weight="bold", fontsize=10)
    handles = [Line2D([], [], color=DEV, lw=1.2, label="Development GEO mean"),
               Line2D([], [], color=EXT, lw=1.2, label="External"),
               Line2D([], [], color="#535E64", marker="D", lw=0, ms=4, label="Development-selected size")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.55, .99), ncol=3, fontsize=6.8)
    save(fig, "图01_三脑区贪婪扩展轨迹与开发增量")


def comparison(summary, ci):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.05), gridspec_kw=dict(left=.09, right=.98, bottom=.20, top=.79, wspace=.35))
    for i, (ax, (region, (_, en))) in enumerate(zip(axes, REGIONS.items())):
        s = summary.loc[summary.region.eq(region)].iloc[0]
        k = int(s.selected_k)
        values = [s.external_base, s.external_selected]
        ax.plot([0, 1], values, color="#B7C0C4", lw=.9)
        for j, (size, color, symbol) in enumerate([(6, DEV, 'o'), (k, EXT, 'D')]):
            q = ci.loc[ci.region.eq(region) & ci.k.eq(size) & ~ci.kind.eq("selected_minus_base")].iloc[0]
            ax.vlines(j, q.lower95, q.upper95, color=color, lw=1.05)
            ax.plot(j, values[j], symbol, color=color, ms=5)
            ax.text(j, min(1.04, q.upper95+.035), f"{values[j]:.3f}", ha="center", fontsize=7)
        ax.set_xlim(-.45, 1.45)
        ax.set_ylim(0, 1.10)
        ax.set_xticks([0, 1], ["Base (6)", f"Selected ({k})"])
        ax.set_yticks([0, .25, .5, .75, 1.0])
        ax.set_title(en, fontsize=8)
        ax.set_ylabel("External AUC" if i == 0 else "")
        ax.tick_params(labelsize=6, length=3)
        ax.text(-.16, 1.14, chr(97+i), transform=ax.transAxes, weight="bold", fontsize=10)
        ax.grid(axis="y", color="#E9EDEF", lw=.4)
    save(fig, "图02_六基因底座与开发选定Panel外部AUC")


def roc_plot(pred, summary):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(left=.08, right=.98, bottom=.19, top=.82, wspace=.34))
    for i, (ax, (region, (_, en))) in enumerate(zip(axes, REGIONS.items())):
        s = summary.loc[summary.region.eq(region)].iloc[0]
        for k, color, label in [(6, DEV, "Base"), (int(s.selected_k), EXT, "Selected")]:
            w = pred.loc[pred.region.eq(region) & pred.k.eq(k)]
            fpr, tpr, _ = roc_curve(w.y, w.score)
            auc = s.external_base if label == "Base" else s.external_selected
            ax.plot(fpr, tpr, color=color, lw=1.1, label=f"{label} ({k}): {auc:.3f}")
        ax.plot([0, 1], [0, 1], color="#BBC3C7", lw=.6, ls=":")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xticks([0, .5, 1])
        ax.set_yticks([0, .5, 1])
        ax.set_xlabel("False-positive rate")
        ax.set_ylabel("True-positive rate" if i == 0 else "")
        ax.set_title(en, fontsize=8)
        ax.tick_params(labelsize=6, length=3)
        ax.legend(loc="lower right", fontsize=6)
        ax.text(-.16, 1.12, chr(97+i), transform=ax.transAxes, weight="bold", fontsize=10)
    save(fig, "图03_底座与扩展Panel外部ROC")


def markdown(df, columns, labels=None):
    lines = ["| "+" | ".join(labels or columns)+" |", "| "+" | ".join("---" for c in columns)+" |"]
    for row in df[columns].itertuples(index=False, name=None):
        lines.append("| "+" | ".join(f"{v:.6f}" if isinstance(v, (float, np.floating)) else str(v) for v in row)+" |")
    return "\n".join(lines)


def report(summary, selected):
    lines = ["# 固定M23＋EDQ：三脑区六基因底座的贪婪扩展", "", "## 主要结果", "",
             "固定额叶GSE261817、颞叶GSE132903、海马GSE48350为外部；颞叶开发已删除GSE159699。采用原六基因底座、M23两成分与EDQ，沿用开发队列等权回代AUC引导的贪婪规则。", "",
             markdown(summary, ["region_cn", "external", "selected_k", "external_base", "external_selected", "external_change", "change_lower95", "change_upper95"],
                 ["脑区", "外部GEO", "开发选定基因数", "六基因外部AUC", "扩展外部AUC", "变化", "差值95%下限", "差值95%上限"]), "",
             "额叶外部AUC提高0.020833；颞叶下降0.074690，海马下降0.009792。额叶和海马差值的配对bootstrap名义95%区间跨0，颞叶区间完全低于0。仅额叶观察到外部点估计上升，没有出现三个脑区一致提升。", "",
             "## 开发拟合表现", "", markdown(summary, ["region_cn", "dev_base", "dev_selected", "dev_change", "added_genes"],
                 ["脑区", "底座开发回代均值", "所选Panel开发回代均值", "开发增量", "新增基因数"]), "",
             "三个脑区的开发训练内指标均明显上升，但未一致转化为外部提升，表现出当前回代目标驱动的扩展可能强化训练内拟合。不能将开发回代均值称为交叉验证Macro或将其接近1解释为外部性能。", "",
             "## 具体做法", "",
             "各脑区使用对应冻结开发集的EDQ排名。六机制各10个名额，第一轮是固定六基因，再按机制顺序轮流选尚未入池的最高分基因补足60个不同候选。多机制基因只占一个位置。第7步起从剩余54个候选逐一试加，每轮只根据各开发GEO回代AUC等权平均选最高者；同分按固定候选顺序。", "",
             "完整走到60基因，保留下降段。在6至60基因中取开发指标峰值，并列取最小规模；额叶39、颞叶32、海马47个。外部全路径在本轮开发路径及规模锁定后计算，外部峰值没有用于重新选Panel。", "",
             "没有内部交叉验证，没有更换模型或评分，也没有改变六机制归属、FDR门槛或验证队列。模型始终保留两成分，从6基因开始，不通过临时改成单成分绘制1基因点。", "",
             "## 基因名单", "", "| 脑区 | 六基因底座 | 新增基因（贪婪顺序） |", "| --- | --- | --- |"]
    for s in summary.itertuples():
        lines.append(f"| {s.region_cn} | {s.base_genes.replace(';', '、')} | {s.added_gene_order.replace(';', '、')} |")
    lines += ["", "每个基因的原EDQ、机制归属、候选池补位槽位、实际添加顺序与是否进入最终Panel，分别见候选证据、完整顺序和开发选定Panel表。最终Panel始终包含原6个基因，没有替换底座。", "",
              "## 图件说明", "",
              "图01：a/c/e依次展示额叶、颞叶、海马的开发回代GEO等权AUC和外部单队列AUC；空心圆为6基因底座，菱形及竖线为开发选定规模，之后路径用虚线。b/d/f为对应每步开发指标变化，负增量照实显示。外部线只是冻结路径的性能展示，未用于确定添加顺序和所选点。", "",
              "图02：每个子图比较该脑区底座与开发选定Panel的外部AUC。误差线为4000次外部病例/对照分层bootstrap百分位95%区间；同一脑区共用抽样以计算配对差值。该区间未作多重比较调整，不包含前期搜索划分或模型历史的不确定性。", "",
              "图03：同一外部供体上，六基因底座与开发选定Panel的ROC。图例括号内为基因数，冒号后为AUC。", "",
              "## 复现与验收", "",
              "保存三个脑区全部4455次候选试加、165个阶段模型、逐步开发各GEO AUC、训练及外部逐供体预测、所有候选与基因顺序、每脑区开发冻结记录和输入校验值。基座外部预测与刚固定的版本一致；所选规模只依赖开发指标；全部AUC和贪婪选择经过独立核对，最终及基座模型重新拟合预测一致。", "",
              "额叶、颞叶、海马分别使用9、4、5个开发GEO，807、138、111条训练记录。海马两个相关GEO保留原来源权重，开发目标仍按照上一轮规则对各GEO等权，不改为家族等权。", "",
              "沿用此前探索后固定的划分，本次结果作为该版本的扩展比较单独保存，没有覆盖六基因四策略结果或手稿。joblib进程退出时可能有ResourceTracker析构警告，模型计算正常完成，所有预测另行验收。", ""]
    (OUT / "结果说明.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    curve = read(R / "三脑区开发与外部完整轨迹.csv")
    summary = read(R / "底座与开发选定Panel外部AUC汇总.csv")
    ci = read(R / "外部AUC及配对增量_Bootstrap区间.csv")
    pred = pd.read_csv(R / "逐步外部供体预测.csv.gz", dtype={"key": str}, float_precision="round_trip")
    path_plot(curve, summary)
    comparison(summary, ci)
    roc_plot(pred, summary)
    report(summary, read(R / "开发选定Panel名单及机制.csv"))
    (QA / "图件自动核对.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved 3 figure sets and Chinese report.")
