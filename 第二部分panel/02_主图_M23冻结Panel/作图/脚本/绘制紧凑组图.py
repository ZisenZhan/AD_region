"""Redraw the three existing evidence blocks as one editable figure."""
from pathlib import Path
import importlib.util
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parent
SOURCES = []
LAYOUT = []
REGIONS = ['frontal_prefrontal', 'temporal_neocortex', 'hippocampus']
TITLES = ['Frontal / prefrontal', 'Temporal neocortex', 'Hippocampus']
CN = ['额叶/前额叶', '颞叶新皮层', '海马']
REGION_COLORS = ['#477CA7', '#B96F91', '#CE8B48']
GEO_COLORS = ['#3278B3', '#42AFC1', '#839F42', '#E3943F', '#9B74B4', '#C8769A', '#9E3D36', '#427C6A', '#8E8376']


def read(path):
    SOURCES.append(path)
    return pd.read_csv(path, float_precision='round_trip')


def save_data(frame, name):
    frame.to_csv(OUT / '源数据' / name, index=False, encoding='utf-8-sig', float_format='%.17g')


def letter(ax, name, x=-.06, y=1.06):
    ax.text(x, y, name, transform=ax.transAxes, fontweight='bold', fontsize=11)


def frame(ax, complete=False):
    ax.tick_params(labelsize=7, width=.65, length=2.7)
    for side, spine in ax.spines.items():
        spine.set_visible(complete or side in ('left', 'bottom'))
        spine.set_linewidth(.7)
        spine.set_color('#30383C')


def area(frame):
    return float(np.trapezoid(frame.TPR, frame.FPR)) if hasattr(np, 'trapezoid') else float(np.trapz(frame.TPR, frame.FPR))


def main():
    for name in ['图件', '源数据', '核对']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    script = BASE / 'M23_EDQ_Panel主文组图_20261116/脚本/组图.py'
    SOURCES.append(script)
    spec = importlib.util.spec_from_file_location('retained_strategy_figure', script)
    strategy_plot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_plot)
    raw = read(BASE / 'M23_EDQ_冻结划分_三脑区贪婪扩展_20261115/输入与划分/冻结版本ABCD外部结果.csv')
    strategy = raw.set_index('region')[list('ABCD')].loc[REGIONS]
    curve = read(BASE / 'M23_EDQ_底座展开交叉验证曲线_20261116/结果/底座展开后完整曲线.csv')
    summary = read(BASE / 'M23_EDQ_三脑区交叉验证贪婪扩展_20261116/结果/三脑区交叉验证贪婪汇总.csv')
    roc_dir = BASE / 'M23_EDQ_Panel交叉验证ROC图集_20261116/源数据'
    individual = read(roc_dir / '全部逐GEO_ROC坐标.csv.gz')
    means = read(roc_dir / '全部等权平均ROC坐标.csv.gz')
    metric = read(roc_dir / '逐队列与规模AUC.csv')
    keep_pairs = {(region, k) for region in REGIONS for k in [6, int(summary.loc[summary.region.eq(region), 'selected_k'].item())]}
    individual = individual.loc[[(r,k) in keep_pairs for r,k in zip(individual.region, individual.k)]]
    means = means.loc[[(r,k) in keep_pairs for r,k in zip(means.region, means.k)]]
    metric = metric.loc[[(r,k) in keep_pairs for r,k in zip(metric.region, metric.k)]]
    save_data(strategy.reset_index(), '四策略六基因外部AUC.csv')
    save_data(curve, '底座展开与扩展完整CV曲线.csv')
    save_data(individual, '逐GEO_ROC坐标.csv.gz')
    save_data(means, '等权平均ROC坐标.csv.gz')
    save_data(metric, '逐GEO_AUC.csv')
    checks = []
    for region in REGIONS:
        s = summary.loc[summary.region.eq(region)].iloc[0]
        for k, expected in [(6, s.cv_base), (int(s.selected_k), s.cv_selected)]:
            points = means.loc[means.region.eq(region) & means.k.eq(k)].sort_values('point')
            aucs = metric.loc[metric.region.eq(region) & metric.k.eq(k)]
            error = max(abs(area(points)-expected), abs(aucs.auc.mean()-expected),
                        abs(curve.loc[curve.region.eq(region) & curve.k.eq(k), 'cv_macro'].item()-expected))
            assert error < 1e-12
            for geo in aucs.geo:
                g = individual.loc[individual.region.eq(region) & individual.k.eq(k) & individual.geo.eq(geo)].sort_values('point')
                actual = aucs.loc[aucs.geo.eq(geo), 'auc'].item()
                assert abs(area(g)-actual) < 1e-12
            checks.append(dict(region=region, k=k, macro_auc=float(expected), max_error=error))

    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False})
    fig = plt.figure(figsize=(11.4, 14.2))
    top = fig.add_gridspec(1, 3, left=.075, right=.985, bottom=.840, top=.949, wspace=.37)
    strategy_axes = []
    for i, region in enumerate(REGIONS):
        ax = fig.add_subplot(top[0, i])
        strategy_plot.bars(ax, strategy.loc[region], TITLES[i], chr(97+i))
        ax.set_title(TITLES[i], fontsize=8.7, loc='left', pad=5)
        ax.tick_params(labelsize=6.7)
        if i:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('External AUC', fontsize=8)
        for text in ax.texts:
            if text.get_text() in ['a', 'b', 'c']:
                text.set_position((-.10, 1.08))
            else:
                text.set_fontsize(7)
        strategy_axes.append(ax)
        LAYOUT.append(dict(panel=chr(97+i), evidence='external AUC', region=region))
    handles = [Patch(facecolor=strategy_plot.COLORS[key], edgecolor='none', label=label)
        for key,label in [('A', 'Region-specific, mechanism-guided'), ('B', 'Region-specific, top-score'),
                          ('C', 'Pooled, mechanism-guided'), ('D', 'Pooled, top-score')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.54, .995), ncol=2,
        fontsize=7.2, handlelength=1.2, columnspacing=2.5, labelspacing=.45,
        frameon=True, fancybox=False, edgecolor='none', facecolor='#EEF0F0')

    tracks = fig.add_gridspec(3, 1, left=.075, right=.985, top=.781, bottom=.416, hspace=.68)
    curve_axes, gene_axes, roc_axes = [], [], []
    low = max(0, np.floor(curve.cv_macro.min()*20)/20-.015)
    high = min(1, np.ceil(curve.cv_macro.max()*20)/20+.015)
    for i, region in enumerate(REGIONS):
        c = curve.loc[curve.region.eq(region)].sort_values('k')
        selected = int(summary.loc[summary.region.eq(region), 'selected_k'].item())
        inner = tracks[i].subgridspec(2, 1, height_ratios=[2.8, 1.], hspace=.11)
        ax, bx = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        curve_axes.append(ax)
        gene_axes.append(bx)
        color = REGION_COLORS[i]
        before, after = c.loc[c.k.le(selected)], c.loc[c.k.ge(selected)]
        ax.plot(before.k, before.cv_macro, color=color, lw=1.35)
        ax.plot(after.k, after.cv_macro, color=color, lw=1., ls=(0,(4,2)))
        ax.scatter([6], [c.loc[c.k.eq(6), 'cv_macro'].item()], facecolors='white', edgecolors=color,
                   linewidths=1., s=22, zorder=4)
        ax.scatter([selected], [c.loc[c.k.eq(selected), 'cv_macro'].item()], color=color,
                   marker='D', s=21, zorder=4)
        ax.set_ylim(low, high)
        ax.set_yticks([.80, .85, .90, .95])
        ax.set_title(TITLES[i], loc='left', fontsize=8.6, pad=4)
        ax.set_ylabel('CV Macro AUC', fontsize=8)
        ax.set_xticks([])
        ax.grid(axis='y', color='#E6EAED', lw=.4)
        increments = c.loc[c.k.gt(2)]
        bx.bar(increments.k, increments.cv_delta, width=.72,
            color=[color if v>=0 else '#ABB3B8' for v in increments.cv_delta], alpha=.78, lw=0)
        bx.axhline(0, color='#667177', lw=.6)
        bx.set_ylabel(r'$\Delta$ AUC', fontsize=7.5)
        labels = [g.replace(' + ', '\n+ ') if k==2 else g for k,g in zip(c.k,c.added_gene)]
        bx.set_xticks(c.k, labels, rotation=90, ha='center')
        bx.tick_params(axis='x', labelsize=6.4, length=2, pad=2)
        bx.tick_params(axis='y', labelsize=6.5, length=2.5)
        for axis in [ax,bx]:
            frame(axis)
            axis.set_xlim(1,61)
            axis.axvline(6, color=color, lw=.6, ls=':', alpha=.6)
            axis.axvline(selected, color='#A4ADB3', lw=.6, ls=':')
        letter(ax, chr(100+i), x=-.038)
        LAYOUT.append(dict(panel=chr(100+i), evidence='development CV curve and increments', region=region))
    fig.legend(handles=[Line2D([], [], color='#59666E', lw=1.3, label='Cross-validated GEO mean'),
        Line2D([], [], color='#59666E', marker='o', mfc='white', lw=0, ms=4.3, label='Six-gene base'),
        Line2D([], [], color='#59666E', marker='D', lw=0, ms=4, label='CV-selected size')],
        loc='upper center', bbox_to_anchor=(.54,.817), ncol=3, fontsize=7.2, columnspacing=2)

    bottom = fig.add_gridspec(2, 3, left=.075, right=.985, bottom=.043, top=.352, wspace=.36, hspace=.39)
    for i, region in enumerate(REGIONS):
        selected = int(summary.loc[summary.region.eq(region), 'selected_k'].item())
        geos = sorted(metric.loc[metric.region.eq(region) & metric.k.eq(6), 'geo'])
        for row,k in enumerate([6,selected]):
            ax = fig.add_subplot(bottom[row,i])
            roc_axes.append(ax)
            ax.set_box_aspect(1)
            ax.plot([0,1],[0,1], color='#DE777C', lw=.55, ls=':')
            for geo,color in zip(geos,GEO_COLORS):
                q = individual.loc[individual.region.eq(region) & individual.k.eq(k) & individual.geo.eq(geo)].sort_values('point')
                auc = metric.loc[metric.region.eq(region) & metric.k.eq(k) & metric.geo.eq(geo), 'auc'].item()
                ax.plot(q.FPR,q.TPR,color=color,lw=.72,label=f'{geo} ({auc:.3f})')
            m = means.loc[means.region.eq(region) & means.k.eq(k)].sort_values('point')
            ax.plot(m.FPR,m.TPR,color='#292C30',lw=1.35,label=f'Mean ROC ({m.macro_auc.iloc[0]:.3f})')
            ax.set_xlim(0,1)
            ax.set_ylim(0,1)
            ax.set_xticks([0,.2,.4,.6,.8,1])
            ax.set_yticks([0,.2,.4,.6,.8,1])
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.1f'))
            ax.set_xlabel('False positive rate',fontsize=7.5)
            ax.set_ylabel('True positive rate' if i==0 else '',fontsize=7.5)
            ax.set_title(f'{TITLES[i]} ({k} genes)',fontsize=8.1,loc='left',pad=5)
            frame(ax,complete=True)
            ax.legend(loc='lower right',fontsize=6.,frameon=True,fancybox=False,facecolor='#E6E8E8',
                edgecolor='none',framealpha=.96,handlelength=1.5,handletextpad=.45,labelspacing=.22,
                borderpad=.35,borderaxespad=.3)
            letter(ax,chr(103+row*3+i),x=-.18)
            LAYOUT.append(dict(panel=chr(103+row*3+i), evidence='development held-out GEO ROC',region=region,k=k))

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside, legend_overflow = [], []
    interblock_gaps = []
    for i, axis in enumerate(gene_axes):
        lowest_gene = min(t.get_window_extent(renderer).y0 for t in axis.get_xticklabels())
        following_title = curve_axes[i+1].title if i<2 else roc_axes[0].title
        # Left-aligned titles are stored separately from the centered title.
        next_axis = curve_axes[i+1] if i<2 else roc_axes[0]
        following_title = next_axis._left_title
        gap = lowest_gene-following_title.get_window_extent(renderer).y1
        interblock_gaps.append(float(gap))
    assert min(interblock_gaps)>2, interblock_gaps
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text():
            b = text.get_window_extent(renderer)
            if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.width+.5 or b.y1>fig.bbox.height+.5:
                outside.append(text.get_text())
    for ax in fig.axes:
        lg=ax.get_legend()
        if lg:
            box,limits=lg.get_window_extent(renderer),ax.get_window_extent(renderer)
            if box.x0<limits.x0-.5 or box.y0<limits.y0-.5 or box.x1>limits.x1+.5 or box.y1>limits.y1+.5:
                legend_overflow.append(ax.get_title())
    assert not outside and not legend_overflow,(outside,legend_overflow)
    name='图01_四策略与贪婪扩展及跨队列ROC紧凑组图'
    for ext in ['png','svg','pdf']:
        fig.savefig(OUT/'图件'/(name+'.'+ext),dpi=420,facecolor='white')
    plt.close(fig)
    assert '<text' in (OUT/'图件'/(name+'.svg')).read_text(encoding='utf-8')
    with Image.open(OUT/'图件'/(name+'.png')) as im:
        pix=np.array(im.convert('RGB'))
        nonwhite=float((pix<245).any(axis=2).mean())
        assert nonwhite>.02 and pix.std()>10
        im.thumbnail((1900,2400))
        im.save(OUT/'核对/组图预览.png')
    audit=dict(status='ok',labeled_panels=12,axes=15,text_outside_canvas=outside,
        legends_outside_axes=legend_overflow,nonwhite_fraction=nonwhite,editable_text=True,
        model_refit=False,external_and_CV_distinguished=True,base_unfolded_from_two_genes=True,
        strategy_axis_lower_bound=.6,numerical_checks=checks,panel_map=LAYOUT)
    audit['gene_labels_to_following_titles_gap_pixels'] = interblock_gaps
    (OUT/'核对/组图验收.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    save_data(pd.DataFrame([dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in SOURCES]),'输入校验.csv')
    lines=['# 四策略、贪婪扩展与跨队列ROC组图','','## 图注','',
        'a-c：额叶/前额叶、颞叶新皮层和海马的六基因四策略对比。纵轴统一0.60-0.92，数值为固定外部验证队列AUC。青绿表示分机制，玫红表示不分机制；浅色为合并脑区，深色为分脑区，连接线对应同一种选基因方法。', '',
        'd-f：同三个脑区的开发交叉验证贪婪路径。各面板上方为逐GEO等权Macro AUC，下方为每步增量及完整添加基因名。采用最新底座展开版，从最佳两基因起点开始，逐步纳入固定六基因；空心点和点线标记六基因处。菱形及灰色点线标记CV选定规模，之后为虚线。灰条为负增量。', '',
        'g-i：三个脑区六基因底座的逐GEO留出ROC；j-l：对应的22、23、19基因扩展Panel。彩线是留出的开发GEO，同一脑区上下两图颜色相同；黑线是GEO等权平均ROC。各图例括号为AUC，Mean ROC括号为Macro AUC。', '',
        '## 数值与评价口径', '',
        '| 脑区 | 六基因CV Macro | 扩展基因数 | 扩展CV Macro |',
        '| --- | --- | --- | --- |']
    for region,cn in zip(REGIONS,CN):
        s=summary.loc[summary.region.eq(region)].iloc[0]
        lines.append(f'| {cn} | {s.cv_base:.6f} | {int(s.selected_k)} | {s.cv_selected:.6f} |')
    lines+=['','a-c与d-l的评价来源不同：前者是单外部队列，后者是开发留来源交叉验证。额叶9个GEO/9折，颞叶4个/4折，海马5个GEO/4个来源折；海马共同来源的两个GEO同时留出、分别算AUC。', '',
        '底座、候选范围固定自开发数据，CV分数参与贪婪和规模选择，因此d-l展示内部开发搜索结果。没有新增模型拟合、显著性检验或置信区间。', '',
        '组图用原始结果和ROC坐标重新绘制，未拼贴截图；第六点与扩展结果保持一致。已核对逐GEO ROC面积、均值ROC面积与Macro，并保存全部源数据。英文图内标注，PNG及可编辑SVG/PDF内容一致。']
    (OUT/'组图说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='ok',figure=str(OUT/'图件'/(name+'.png')),nonwhite=nonwhite),ensure_ascii=False))


if __name__=='__main__':
    main()
