"""Compact, fixed-row model heatmaps; rankings remain separate source tables."""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

OUT = Path(__file__).resolve().parents[1]
DATA = OUT / "结果"
FIG = OUT / "图件"
SOURCE = FIG / "热图源数据"
for p in (FIG, SOURCE):
    p.mkdir(exist_ok=True)
REGIONS = {
    "frontal_prefrontal": ("额叶前额叶", "Frontal / prefrontal cortex"),
    "temporal_neocortex": ("颞叶新皮层", "Temporal neocortex"),
    "hippocampus": ("海马", "Hippocampus"),
}
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 6.5, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "axes.linewidth": .6, "axes.unicode_minus": False})
CMAP = LinearSegmentedColormap.from_list("auc", ["#376A88", "#ADC6D2", "#F7F7F4", "#DFA8A3", "#A34246"])
NORM = Normalize(0, 1)
TEXT = "#20272D"
models = pd.read_csv(OUT / "输入与划分/36模型配置.csv").sort_values("row_order")
internal = pd.read_csv(DATA / "internal_逐GEO_AUC.csv", float_precision="round_trip")
table = pd.read_csv(DATA / "36配置_三脑区四策略完整结果.csv", float_precision="round_trip")
quality = []


def short_label(row):
    p = json.loads(row.parameters)
    family = row.family
    if family.startswith("logistic_") or family == "linear_svm":
        name = {"logistic_l2": "L2 logistic", "logistic_l1": "L1 logistic", "logistic_elasticnet": "Elastic net", "linear_svm": "Linear SVM"}[family]
        value = f"C={p['C']:g}"
        if "l1_ratio" in p:
            value += f", l1={p['l1_ratio']:g}"
    elif family == "rbf_svm":
        name, value = "RBF SVM", f"C={p['C']:g}, gamma={p['gamma']}"
    elif family == "shrinkage_lda":
        name, value = "LDA", f"shrinkage={p['shrinkage']}"
    elif family == "gaussian_nb":
        name, value = "Gaussian NB", f"var={p['var_smoothing']:.0e}"
    elif family == "pls_logistic":
        name, value = "PLS-logistic", f"n={p['n_components']}"
    elif family == "random_forest":
        name, value = "RF", f"depth={p['max_depth']}, leaf={p['min_samples_leaf']}"
    else:
        name = {"gradient_boosting": "GBM", "adaboost": "AdaBoost", "catboost": "CatBoost"}[family]
        depth = p.get("depth", p.get("max_depth"))
        count = p.get("iterations", p.get("n_estimators"))
        value = f"d={depth}, n={count}, lr={p['learning_rate']:g}"
    return f"{row.model}  {name}  {value}"


LABELS = [short_label(r) for r in models.itertuples()]
BOUNDARIES = np.where(models.family.to_numpy()[1:] != models.family.to_numpy()[:-1])[0] + 1


def cells(ax, data, headers, ylabels=False):
    values = np.asarray(data, dtype=float)
    image = ax.imshow(np.ma.masked_invalid(values), cmap=CMAP, norm=NORM, aspect="auto", interpolation="none")
    ax.set_xticks(np.arange(values.shape[1]), headers)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=4, labelsize=6.2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(55)
        tick.set_ha("left")
        tick.set_rotation_mode("anchor")
    ax.set_yticks(np.arange(36), LABELS if ylabels else [])
    ax.tick_params(axis="y", length=0, pad=6, labelsize=6.1, colors=TEXT)
    ax.set_xticks(np.arange(-.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, 36, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.45)
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for boundary in BOUNDARIES:
        ax.axhline(boundary-.5, color="#AEB6BB", lw=.6, zorder=5)
    for i in range(36):
        for j in range(values.shape[1]):
            value = values[i, j]
            label = f"{value:.3f}" if np.isfinite(value) else "NA"
            color = "white" if np.isfinite(value) and (value > .90 or value < .16) else TEXT
            ax.text(j, i, label, ha="center", va="center", color=color, fontsize=5.8)
    return image


def finish(fig, name, matrix):
    color_ax = fig.add_axes([.46, .04, .30, .012])
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=NORM, cmap=CMAP), cax=color_ax, orientation="horizontal", ticks=[0, .25, .5, .75, 1])
    cb.set_label("ROC AUC", fontsize=7, labelpad=3)
    cb.ax.tick_params(labelsize=6, length=2, width=.5)
    cb.outline.set_linewidth(.5)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(FIG / (name+"."+extension), dpi=400, facecolor="white")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for item in fig.findobj(mpl.text.Text):
        if not item.get_visible() or not item.get_text():
            continue
        box = item.get_window_extent(renderer)
        if box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width+.5 or box.y1 > fig.bbox.height+.5:
            outside.append(item.get_text())
    assert not outside, (name, outside)
    rgba = np.asarray(fig.canvas.buffer_rgba())
    quality.append(dict(figure=name, width_inches=fig.get_figwidth(), height_inches=fig.get_figheight(),
                        matrix_rows=matrix.shape[0], matrix_columns=matrix.shape[1], missing_cells=int(matrix.isna().sum().sum()),
                        text_outside_canvas=outside, nonwhite_fraction=float((rgba[:, :, :3] < 245).any(axis=2).mean()),
                        svg_editable_text="<text" in (FIG / (name+".svg")).read_text(encoding="utf-8")))
    plt.close(fig)


def main_A(region, number):
    chinese, english = REGIONS[region]
    z = internal.loc[internal.region.eq(region) & internal.arm.eq("A")]
    geos = sorted(z.geo.unique())
    folded = z.pivot(index="model", columns="geo", values="auc").reindex(index=models.model, columns=geos)
    aggregate = table.loc[table.region.eq(region) & table.arm.eq("A")].set_index("model").reindex(models.model)
    matrix = folded.copy()
    matrix["Dev macro"] = aggregate.internal_macro
    ext = aggregate.external_GEO.iloc[0]
    matrix["External "+ext] = aggregate.external_auc
    matrix.to_csv(SOURCE / (chinese+"_六机制底座A.csv"), encoding="utf-8-sig", float_format="%.17g")
    fig = plt.figure(figsize=(7.2, 7.8))
    widths = [len(geos), 1, 1]
    grid = fig.add_gridspec(1, 3, left=.333, right=.985, bottom=.105, top=.847,
                           wspace=.12, width_ratios=widths)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    cells(axes[0], folded, geos, ylabels=True)
    cells(axes[1], aggregate[["internal_macro"]], ["Dev macro"])
    cells(axes[2], aggregate[["external_auc"]], ["External\n"+ext])
    fig.text(.333, .965, english, weight="bold", fontsize=10, color=TEXT, ha="left")
    fig.text(.012, .965, chr(97+number), fontsize=11, weight="bold", color=TEXT)
    for axis in axes[1:]:
        axis.add_patch(Rectangle((-.5, -.5), 1, 36, fill=False, edgecolor="#747E85", lw=.6, clip_on=False))
    finish(fig, f"图0{number+1}_{chinese}_六机制底座36模型热力图", matrix)


def supplement(region, number):
    chinese, english = REGIONS[region]
    matrix = pd.DataFrame(index=models.model)
    for arm in "ABCD":
        z = table.loc[table.region.eq(region) & table.arm.eq(arm)].set_index("model").reindex(models.model)
        matrix[arm+" Dev"] = z.internal_macro
        matrix[arm+" External"] = z.external_auc
    matrix.to_csv(SOURCE / (chinese+"_四策略.csv"), encoding="utf-8-sig", float_format="%.17g")
    fig = plt.figure(figsize=(7.2, 7.8))
    grid = fig.add_gridspec(1, 4, left=.333, right=.985, bottom=.105, top=.847, wspace=.15)
    for i, arm in enumerate("ABCD"):
        axis = fig.add_subplot(grid[0, i])
        cells(axis, matrix.iloc[:, i*2:i*2+2], ["Dev", "External"], ylabels=i == 0)
        axis.text(.5, 1.06, arm, transform=axis.transAxes, fontsize=9, weight="bold", ha="center")
    fig.text(.333, .965, english, weight="bold", fontsize=10, color=TEXT)
    finish(fig, f"补充图0{number+1}_{chinese}_四策略36模型热力图", matrix)


if __name__ == "__main__":
    for i, region in enumerate(REGIONS):
        main_A(region, i)
        supplement(region, i)
    (OUT / "核对/图件自动检查.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved 3 main and 3 supplementary heatmaps, each as PNG/SVG/PDF.")
