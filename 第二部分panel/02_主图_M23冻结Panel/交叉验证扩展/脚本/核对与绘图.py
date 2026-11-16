"""Verify all saved CV metrics and render the frozen forward path."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

OUT = Path(__file__).resolve().parents[1]
R, FIG, QA = [OUT / n for n in ["结果", "图件", "核对"]]
REGIONS = {"frontal_prefrontal": ("额叶/前额叶", "Frontal / prefrontal cortex", "#477CA7"),
           "temporal_neocortex": ("颞叶新皮层", "Temporal neocortex", "#B96F91"),
           "hippocampus": ("海马", "Hippocampus", "#CE8B48")}


def read(name, **kwargs):
    return pd.read_csv(R / name, float_precision="round_trip", **kwargs)


def verify():
    curve = read("交叉验证完整轨迹.csv")
    trial = read("全部候选试加.csv.gz")
    per_geo = read("逐步逐GEO_AUC.csv")
    pred = read("逐步OOF预测.csv.gz", dtype={"key": str})
    summary = read("三脑区交叉验证贪婪汇总.csv")
    candidates = pd.read_csv(OUT / "输入与划分/固定60候选及机制与EDQ证据.csv", float_precision="round_trip")
    assert len(curve) == 165 and len(trial) == 4455 and len(per_geo) == 55 * (9 + 4 + 5)
    assert not pred.duplicated(["region", "k", "geo", "key"]).any()
    errors, selections = [], []
    for region in REGIONS:
        c = curve.loc[curve.region.eq(region)].sort_values("k")
        s = summary.loc[summary.region.eq(region)].iloc[0]
        assert c.k.tolist() == list(range(6, 61))
        initial = candidates.loc[candidates.region.eq(region)].sort_values("candidate_order").gene.tolist()
        assert len(set(initial)) == 60 and c.iloc[0].genes.split(";") == initial[:6]
        path = initial[:6]
        for row in c.itertuples():
            if row.k > 6:
                z = trial.loc[trial.region.eq(region) & trial.k.eq(row.k)]
                remaining = [g for g in initial if g not in path]
                assert z.candidate.tolist() == remaining
                best_value, chosen = -np.inf, None
                for t in z.itertuples():
                    values = np.array([getattr(t, n) for n in z.columns if n.startswith("auc_")], dtype=float)
                    values = values[np.isfinite(values)]
                    assert len(values) == s.cv_GEOs
                    assert abs(t.cv_macro-values.mean()) < 1e-12
                    if t.cv_macro > best_value+1e-12:
                        best_value, chosen = t.cv_macro, t.candidate
                assert chosen == row.added_gene and z.loc[z.selected, "candidate"].tolist() == [chosen]
                path.append(chosen)
                assert abs(best_value-row.cv_macro) < 1e-12
                selections.append(dict(region=region, k=row.k, selected=chosen, correct=True))
            assert row.genes.split(";") == path and len(path) == len(set(path))
            assert set(initial[:6]) <= set(path)
            z = per_geo.loc[per_geo.region.eq(region) & per_geo.k.eq(row.k)]
            assert len(z) == s.cv_GEOs and abs(z.auc.mean()-row.cv_macro) < 1e-12
            block = pred.loc[pred.region.eq(region) & pred.k.eq(row.k)]
            for geo, q in block.groupby("geo"):
                ref = pred.loc[pred.region.eq(region) & pred.k.eq(6) & pred.geo.eq(geo), ["key", "y"]].sort_values("key").reset_index(drop=True)
                assert q[["key", "y"]].sort_values("key").reset_index(drop=True).equals(ref)
                auc = roc_auc_score(q.y, q.score)
                p, n = q.loc[q.y.eq(1), "score"].to_numpy()[:, None], q.loc[q.y.eq(0), "score"].to_numpy()
                pairwise = ((p > n)+.5*(p == n)).mean()
                expected = z.loc[z.geo.eq(geo), "auc"].item()
                assert abs(auc-expected) < 1e-12 and abs(pairwise-expected) < 1e-12
                errors.append(abs(auc-expected))
        peak = c.cv_macro.max()
        expected_k = c.loc[c.cv_macro.ge(peak-1e-12), "k"].iloc[0]
        assert expected_k == s.selected_k and abs(peak-s.cv_selected) < 1e-12
        assert abs(s.cv_base-c.iloc[0].cv_macro) < 1e-12
        assert s.genes.split(";") == path[:int(s.selected_k)]
        folds = pd.read_csv(OUT / "输入与划分" / (region+"_交叉验证划分.csv"))
        assert folds.fold.nunique() == s.cv_folds and len(folds) == s.cv_GEOs
        for f, group in folds.groupby("fold"):
            test = set(group.test_GEO)
            assert all(not test & set(t.split(";")) for t in group.train_GEOs)
            if f == "Hisayama":
                assert test == ({"GSE173954", "GSE36980"} if region == "hippocampus" else {"GSE36980"})
    pipeline = read("完整流水线独立核对.csv")
    assert pipeline.auc_identical.all() and pipeline.prediction_max_error.max() < 1e-10
    pd.DataFrame(selections).to_csv(QA / "逐步贪婪选择独立核查.csv", index=False, encoding="utf-8-sig")
    report = dict(status="VERIFIED", path_points=len(curve), candidate_trials=len(trial),
        OOF_rows=len(pred), GEO_auc_cells=len(per_geo), max_auc_recalculation_error=max(errors),
        max_cached_vs_full_pipeline_prediction_error=float(pipeline.prediction_max_error.max()),
        source_blocked_folds=dict(zip(summary.region, summary.cv_folds.astype(int))),
        reads_external_expression_or_labels=False, nested_feature_selection=False,
        selection_uses_same_CV_scores=True, source_pool_fixed_from_full_development=True)
    (QA / "独立数值验收.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return curve, summary


def plot(curve, summary):
    FIG.mkdir(exist_ok=True)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False})
    fig = plt.figure(figsize=(13.0, 10.3))
    grids = fig.add_gridspec(3, 1, left=.075, right=.985, top=.945, bottom=.13, hspace=.87)
    low = max(0., np.floor(curve.cv_macro.min()*20)/20-.015)
    high = min(1., np.ceil(curve.cv_macro.max()*20)/20+.015)
    for i, (region, (_, title, color)) in enumerate(REGIONS.items()):
        c = curve.loc[curve.region.eq(region)].sort_values("k")
        selected = int(summary.loc[summary.region.eq(region), "selected_k"].item())
        grid = grids[i, 0].subgridspec(2, 1, height_ratios=[3.1, 1], hspace=.13)
        ax, bx = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[1, 0])
        a, b = c.loc[c.k.le(selected)], c.loc[c.k.ge(selected)]
        ax.plot(a.k, a.cv_macro, color=color, lw=1.5)
        ax.plot(b.k, b.cv_macro, color=color, lw=1.2, ls=(0, (4, 2)))
        ax.scatter([6], [c.iloc[0].cv_macro], s=26, facecolor="white", edgecolor=color, linewidth=1.1, zorder=5)
        peak = c.loc[c.k.eq(selected)].iloc[0]
        ax.scatter([selected], [peak.cv_macro], marker="D", color=color, s=25, zorder=5)
        ax.axvline(selected, color="#A9B1B5", lw=.7, ls=":")
        ax.set_xlim(5, 61)
        ax.set_ylim(low, high)
        ax.set_ylabel("CV Macro AUC", fontsize=9)
        ax.set_xticks([6, 10, 20, 30, 40, 50, 60], [])
        ax.set_title(title, fontsize=10, loc="left", pad=6)
        ax.grid(axis="y", lw=.45, color="#E6EAED")
        ax.tick_params(labelsize=8, length=3)
        ax.text(-.057, 1.06, chr(97+i*2), fontsize=11, fontweight="bold", transform=ax.transAxes)
        increments = c.loc[c.k.gt(6)]
        bx.bar(increments.k, increments.cv_delta, width=.72,
            color=[color if v >= 0 else "#A6AEB3" for v in increments.cv_delta], alpha=.75, linewidth=0)
        bx.axhline(0, color="#5C666C", lw=.65)
        bx.axvline(selected, color="#A9B1B5", lw=.7, ls=":")
        bx.set_xlim(5, 61)
        bx.set_xticks(c.k, ["Base (6)" if k == 6 else g for k, g in zip(c.k, c.added_gene)], rotation=90, ha="center")
        bx.set_ylabel(r"$\Delta$ Macro AUC", fontsize=8)
        bx.tick_params(axis="x", labelsize=6.3, length=2)
        bx.tick_params(axis="y", labelsize=7, length=3)
        bx.text(-.057, 1.01, chr(98+i*2), fontsize=11, fontweight="bold", transform=bx.transAxes)
    legend = [Line2D([], [], color="#59666E", lw=1.5, label="Cross-validated GEO mean"),
        Line2D([], [], color="#59666E", marker="o", mfc="white", lw=0, ms=5, label="Six-gene base"),
        Line2D([], [], color="#59666E", marker="D", lw=0, ms=4, label="CV-selected size")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.54, .998), ncol=3, fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width+.5 or box.y1 > fig.bbox.height+.5:
                outside.append(text.get_text())
    assert not outside, outside
    name = "图01_三脑区交叉验证贪婪曲线与逐基因增量"
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(FIG / (name+"."+ext), dpi=400, facecolor="white")
    plt.close(fig)
    with Image.open(FIG / (name+".png")) as im:
        pixels = np.asarray(im.convert("RGB"))
        assert pixels.std() > 10 and (pixels < 245).any(axis=2).mean() > .01
        dimensions = im.size
    svg = (FIG / (name+".svg")).read_text(encoding="utf-8")
    assert "<text" in svg and "External AUC" not in svg
    (QA / "图件验收.json").write_text(json.dumps(dict(text_outside_canvas=outside, png_dimensions=dimensions,
        nonwhite_fraction=float((pixels<245).any(axis=2).mean()), editable_svg=True, preserves_declines=True,
        uses_real_CV_scores=True, uses_external_scores=False), ensure_ascii=False, indent=2), encoding="utf-8")


def write_report(curve, summary):
    lines = ["# M23＋EDQ：仅开发交叉验证的贪婪扩展", "", "## 结果", "",
        "| 脑区 | 开发GEO / 来源折数 | 六基因CV Macro AUC | CV选定基因数 | CV峰值 | 变化 |",
        "| --- | --- | --- | --- | --- | --- |"]
    for region, (cn, _, _) in REGIONS.items():
        z = summary.loc[summary.region.eq(region)].iloc[0]
        lines.append(f"| {cn} | {int(z.cv_GEOs)} / {int(z.cv_folds)} | {z.cv_base:.6f} | {int(z.selected_k)} | {z.cv_selected:.6f} | {z.cv_change:+.6f} |")
    lines += ["", "本轮仅计算开发队列留一来源家族交叉验证，未读取外部队列表达、标签或预测，没有外部AUC。Macro是逐GEO留出AUC的等权平均，不含训练回代。", "",
        "## 候选池", "", "沿用当前六机制与EDQ：每机制10个名额，固定机制顺序轮流从排名中取尚未入池的最高基因，去重补位，合计60个不同候选。前6个是已冻结底座，另有54个可以逐步试加。本轮不改评分、不改模型、不改注释，也不新增FDR硬门槛。", "",
        "## CV选定Panel", ""]
    for region, (cn, _, _) in REGIONS.items():
        z = summary.loc[summary.region.eq(region)].iloc[0]
        genes = z.genes.split(";")
        lines += [f"### {cn}", "", "六基因底座："+"、".join(genes[:6])+"。", "",
            "新增基因（实际贪婪顺序）："+("、".join(genes[6:]) if len(genes)>6 else "无")+"。", ""]
    lines += ["## 交叉验证与搜索", "",
        "额叶9个GEO为9折，颞叶4个为4折；海马5个GEO为4个来源折，GSE36980与GSE173954共同留出，但分别计算AUC，所以海马Macro仍对5个GEO等权。每折仅在剩余开发记录拟合中位数填补、标准化、PLS和Logistic；原来源权重的频数实现不变。供体有多个测试记录时按亚区内、再跨可用亚区平均预测。", "",
        "每轮尝试加入所有剩余候选，按CV Macro最高者添加，同分依冻结候选顺序。完整路径从6走到60，下降段保留；在该范围内取CV峰值，并列选最小规模。三个区域共4455次候选试加，全部记录已保存。", "",
        "## 图件", "",
        "图01中a/c/e为额叶、颞叶和海马的开发交叉验证Macro曲线；空心点是六基因底座，菱形及竖线标记CV选定规模，之后路径为虚线。b/d/f为每步CV Macro变化，灰条表示下降。横轴逐一列新增基因，Base (6)表示固定的六基因集合。没有置信带或显著性检验。", "",
        "## 解释与核对", "",
        "六基因底座与候选池固定自完整开发集，且同一套CV分数反复用于贪婪及规模选择，所以峰值反映本次内部搜索目标的优化程度，不是全流程嵌套验证或无偏外部性能。", "",
        "每一步逐GEO AUC均从保存的OOF预测与病例—对照两两比较复算；全部贪婪选择及峰值规模独立核对。底座、所选规模及60基因模型均逐折用完整原流水线重拟合，缓存预测与完整流水线预测一致。原结果与手稿未改。"]
    (OUT / "结果说明.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    curve, summary = verify()
    plot(curve, summary)
    write_report(curve, summary)
    print(summary[["region", "cv_base", "selected_k", "cv_selected", "cv_change"]].to_string(index=False))
