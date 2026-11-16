"""Unfold the fixed base while preserving the existing post-six path."""
from pathlib import Path
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
import json
import hashlib
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.metrics import roc_auc_score
from PIL import Image
from threadpoolctl import threadpool_limits

OUT = Path(__file__).resolve().parents[1]
BASE = OUT.parent
OLD = BASE / 'M23_EDQ_三脑区交叉验证贪婪扩展_20261116'
PREV = BASE / 'M23_EDQ_冻结划分_三脑区贪婪扩展_20261115'
DB = BASE / '别名审计与评分轮换_1107/训练与评测缓存'
sys.path.insert(0, str(OLD / '脚本'))
from 模型与折 import Fold

REGIONS = {
    'frontal_prefrontal': ('额叶/前额叶', 'Frontal / prefrontal cortex', '#477CA7'),
    'temporal_neocortex': ('颞叶新皮层', 'Temporal neocortex', '#B96F91'),
    'hippocampus': ('海马', 'Hippocampus', '#CE8B48'),
}
READS = set()


def read(path, pickle=False):
    READS.add(path)
    return pd.read_pickle(path) if pickle else pd.read_csv(path, float_precision='round_trip', dtype={'key': str})


def put(rows, name):
    pd.DataFrame(rows).to_csv(OUT / '结果' / name, index=False, encoding='utf-8-sig', float_format='%.17g')


def source(geo):
    if geo in {'GSE36980', 'GSE173954'}:
        return 'Hisayama'
    if geo in {'GSE33000', 'GSE44768', 'GSE44770', 'GSE44771'}:
        return 'HBTRC'
    return geo


def compute():
    original = read(OLD / '结果/交叉验证完整轨迹.csv')
    oldpred = read(OLD / '结果/逐步OOF预测.csv.gz')
    summary = read(OLD / '结果/三脑区交叉验证贪婪汇总.csv')
    rows, trials, predictions, aucs, checks, orders = [], [], [], [], [], []
    for region in REGIONS:
        p = read(PREV / '运行缓存' / (region + '_输入.pkl'), pickle=True)
        genes, meta, x = p['base'], p['meta'], p['train']
        assert len(genes) == len(set(genes)) == 6
        assert p['external'] not in set(meta.gse)
        groups = {}
        for uid in p['units']:
            groups.setdefault(source(uid.split('__')[1]), []).append(uid)
        folds = []
        for family, units in sorted(groups.items()):
            held = {u.split('__')[1] for u in units}
            keep = ~meta.gse.isin(held)
            assert not {source(g) for g in meta.loc[keep, 'gse']} & {source(g) for g in held}
            tests = {}
            for uid in units:
                geo = uid.split('__')[1]
                assert geo != p['external']
                xt = read(DB / (uid + '_test.pkl'), pickle=True)[genes]
                md = read(DB / (uid + '_md.pkl'), pickle=True)
                tests[geo] = (xt, md)
            folds.append(Fold(family, x.loc[keep, genes], meta.loc[keep], tests))

        def evaluate(panel):
            values = {}
            for fold in folds:
                one = fold.evaluate(panel)
                assert not set(values) & set(one)
                values.update(one)
            return float(np.mean([z['auc'] for z in values.values()])), values

        def choose(candidates):
            best, chosen, result, start = -np.inf, None, None, len(trials)
            for panel in candidates:
                value, values = evaluate(panel)
                trials.append(dict(region=region, k=len(panel), genes=';'.join(panel), cv_macro=value))
                if value > best + 1e-12:
                    best, chosen, result = value, list(panel), values
            for trial in trials[start:]:
                trial['selected'] = trial['genes'] == ';'.join(chosen)
            return chosen, best, result

        def commit(panel, value, values):
            k = len(panel)
            previous = next((z['cv_macro'] for z in reversed(rows) if z['region'] == region), np.nan)
            label = ' + '.join(panel) if k == 2 else panel[-1]
            rows.append(dict(region=region, k=k, added_gene=label, cv_macro=value,
                             cv_delta=value-previous, genes=';'.join(panel)))
            for fold in folds:
                for geo, query in fold.queries.items():
                    score, y = values[geo]['score'], query['y']
                    actual = roc_auc_score(y, score)
                    assert abs(actual-values[geo]['auc']) < 1e-12
                    aucs.append(dict(region=region, k=k, fold=fold.fold_id, geo=geo,
                                     auc=actual, n=len(y), AD=int(y.sum()), control=int((1-y).sum())))
                    predictions.extend(dict(region=region, k=k, fold=fold.fold_id, geo=geo,
                                            key=str(key), y=int(yy), score=float(s))
                                       for key, yy, s in zip(query['keys'], y, score))

        panel, value, values = choose(itertools.combinations(genes, 2))
        commit(panel, value, values)
        for k in range(3, 7):
            panel, value, values = choose([panel + [g] for g in genes if g not in panel])
            commit(panel, value, values)
        assert set(panel) == set(genes)
        for pos, gene in enumerate(panel, 1):
            orders.append(dict(region=region, position=pos, gene=gene, first_point_contains_two=pos<=2))
        expected = original.loc[original.region.eq(region) & original.k.eq(6)].iloc[0]
        assert abs(value-expected.cv_macro) < 1e-12
        for fold in folds:
            _, reference = fold.reference(panel)
            for geo, score in reference.items():
                q = fold.queries[geo]
                frozen = oldpred.loc[oldpred.region.eq(region) & oldpred.k.eq(6) & oldpred.geo.eq(geo)].set_index('key').loc[q['keys']]
                error = float(np.max(np.abs(score-frozen.score.to_numpy())))
                cache_error = float(np.max(np.abs(score-values[geo]['score'])))
                assert error < 1e-10 and cache_error < 1e-10
                assert np.array_equal(q['y'], frozen.y)
                checks.append(dict(region=region, geo=geo, k6_old_prediction_error=error,
                                   k6_full_pipeline_error=cache_error, k6_macro_error=abs(value-expected.cv_macro)))
        print(region, 'base order', panel, 'k6', value, flush=True)

    basecurve = pd.DataFrame(rows)
    curve = pd.concat([basecurve, original.loc[original.k.gt(6)]], ignore_index=True).sort_values(['region', 'k'])
    for region in REGIONS:
        assert curve.loc[curve.region.eq(region), 'k'].tolist() == list(range(2, 61))
    later = curve.loc[curve.k.gt(6)].reset_index(drop=True)
    frozen_later = original.loc[original.k.gt(6)].sort_values(['region', 'k']).reset_index(drop=True)
    pd.testing.assert_frame_equal(later, frozen_later, check_exact=True)
    put(trials, '底座全部候选组合评分.csv')
    put(orders, '底座内部贪婪顺序.csv')
    put(curve, '底座展开后完整曲线.csv')
    put(pd.concat([pd.DataFrame(predictions), oldpred.loc[oldpred.k.gt(6)]], ignore_index=True), '完整逐步OOF预测.csv.gz')
    oldaucs = read(OLD / '结果/逐步逐GEO_AUC.csv')
    fullaucs = pd.concat([pd.DataFrame(aucs), oldaucs.loc[oldaucs.k.gt(6)]], ignore_index=True)
    put(fullaucs, '完整逐步逐GEO_AUC.csv')
    macro = fullaucs.groupby(['region', 'k']).auc.mean()
    errors = [abs(row.cv_macro-macro.loc[(row.region, row.k)]) for row in curve.itertuples()]
    assert max(errors) < 1e-12
    put(checks, '第六点与原流水线核对.csv')
    report = dict(status='ok', trial_panels=len(trials), first_k=2, components=2,
                  later_path_unchanged=True, macro_recalculation_max_error=max(errors), uses_external=False,
                  nested_feature_selection=False, k6_prediction_max_error=max(z['k6_old_prediction_error'] for z in checks))
    (OUT / '核对/数值验收.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    manifest = []
    for path in sorted(READS):
        with path.open('rb') as f:
            manifest.append(dict(path=str(path), sha256=hashlib.file_digest(f, 'sha256').hexdigest()))
    put(manifest, '读取输入校验.csv')
    return curve, summary, pd.DataFrame(orders)


def plot(curve, summary):
    mpl.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .7,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'legend.frameon': False})
    fig = plt.figure(figsize=(14.0, 10.8))
    grids = fig.add_gridspec(3, 1, left=.075, right=.985, top=.945, bottom=.135, hspace=.9)
    low = max(0, np.floor(curve.cv_macro.min()*20)/20-.015)
    high = min(1, np.ceil(curve.cv_macro.max()*20)/20+.015)
    for i, (region, (_, title, color)) in enumerate(REGIONS.items()):
        c = curve.loc[curve.region.eq(region)].sort_values('k')
        selected = int(summary.loc[summary.region.eq(region), 'selected_k'].item())
        grid = grids[i].subgridspec(2, 1, height_ratios=[3.1, 1], hspace=.14)
        ax, bx = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
        a, b = c.loc[c.k.le(selected)], c.loc[c.k.ge(selected)]
        ax.plot(a.k, a.cv_macro, color=color, lw=1.5)
        ax.plot(b.k, b.cv_macro, color=color, lw=1.2, ls=(0, (4, 2)))
        base = c.loc[c.k.eq(6)].iloc[0]
        ax.scatter([6], [base.cv_macro], s=27, facecolor='white', edgecolor=color, linewidth=1.1, zorder=5)
        peak = c.loc[c.k.eq(selected)].iloc[0]
        ax.scatter([selected], [peak.cv_macro], marker='D', color=color, s=25, zorder=5)
        ax.set_ylim(low, high)
        ax.set_ylabel('CV Macro AUC', fontsize=9)
        ax.set_title(title, fontsize=10, loc='left', pad=6)
        ax.grid(axis='y', lw=.45, color='#E6EAED')
        ax.set_xticks([2, 6, 10, 20, 30, 40, 50, 60], [])
        ax.tick_params(labelsize=8, length=3)
        increments = c.loc[c.k.gt(2)]
        bx.bar(increments.k, increments.cv_delta, width=.72,
               color=[color if z>=0 else '#A6AEB3' for z in increments.cv_delta], alpha=.75, linewidth=0)
        bx.axhline(0, color='#5C666C', lw=.65)
        for axis in (ax, bx):
            axis.set_xlim(1, 61)
            axis.axvline(6, color=color, lw=.65, ls=':', alpha=.6)
            axis.axvline(selected, color='#A9B1B5', lw=.7, ls=':')
        labels = [g.replace(' + ', '\n+ ') if k==2 else g for k,g in zip(c.k,c.added_gene)]
        bx.set_xticks(c.k, labels, rotation=90, ha='center')
        bx.tick_params(axis='x', labelsize=6.4, length=2)
        bx.tick_params(axis='y', labelsize=7, length=3)
        bx.set_ylabel(r'$\Delta$ Macro AUC', fontsize=8)
        for axis, letter in [(ax, chr(97+2*i)), (bx, chr(98+2*i))]:
            axis.text(-.057, 1.06, letter, transform=axis.transAxes, fontsize=11, fontweight='bold')
    fig.legend(handles=[Line2D([], [], color='#59666E', lw=1.5, label='Cross-validated GEO mean'),
        Line2D([], [], color='#59666E', marker='o', mfc='white', lw=0, ms=5, label='Six-gene base'),
        Line2D([], [], color='#59666E', marker='D', lw=0, ms=4, label='CV-selected size')],
        loc='upper center', bbox_to_anchor=(.54, .998), ncol=3, fontsize=8)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    outside = []
    for text in fig.findobj(mpl.text.Text):
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.width+.5 or box.y1>fig.bbox.height+.5:
                outside.append(text.get_text())
    assert not outside, outside
    name = '图01_底座展开的三脑区交叉验证贪婪曲线'
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(OUT / '图件' / (name+'.'+ext), dpi=400, facecolor='white')
    plt.close(fig)
    with Image.open(OUT / '图件' / (name+'.png')) as im:
        pixels = np.array(im.convert('RGB'))
        assert pixels.std()>10 and (pixels<245).any(axis=2).mean()>.01
        im.thumbnail((1800, 1800))
        im.save(OUT / '核对/图件预览.png')
    svg = (OUT / '图件' / (name+'.svg')).read_text(encoding='utf-8')
    assert '<text' in svg
    (OUT / '核对/图件验收.json').write_text(json.dumps(dict(outside=outside, editable_text=True,
        first_point_two_genes=True, six_gene_marker=True, after_peak_dashed=True)), encoding='utf-8')


def report(curve, summary, orders):
    lines = ['# 六基因底座展开结果', '',
        '固定原六基因成员，在开发交叉验证中选出最佳两基因起点，再逐个加入剩余四个底座基因；到第六点后接上原有扩展路径。', '',
        'M23固定两个PLS成分，不能以相同配置拟合单基因，所以第一点是两个基因，横轴一并标注。第六点及后续全部结果不变；没有重新选择外部队列或读取外部结果。', '',
        '| 脑区 | 底座累积顺序（前两项共同构成起点） | 两基因CV Macro | 六基因CV Macro | 原CV峰值 |',
        '| --- | --- | --- | --- | --- |']
    for region, (cn, _, _) in REGIONS.items():
        c = curve.loc[curve.region.eq(region)].sort_values('k')
        order = orders.loc[orders.region.eq(region)].sort_values('position').gene.tolist()
        s = summary.loc[summary.region.eq(region)].iloc[0]
        lines.append(f"| {cn} | {' → '.join(order)} | {c.iloc[0].cv_macro:.6f} | {s.cv_base:.6f} | {s.cv_selected:.6f}（{int(s.selected_k)}基因） |")
    lines += ['', '## 图的读法', '',
        'a/c/e为三个脑区的CV Macro曲线；b/d/f为每步加入一个基因后的变化。空心点及其竖直点线标记六基因底座；菱形及其竖直点线标记原CV选定规模。峰值后的路径为虚线，下降段如实保留。首个两基因起点没有增量条，其后均为新增一个基因的增量。', '',
        'Macro为开发GEO留出AUC的等权平均。额叶9个GEO/9折、颞叶4个/4折、海马5个/4个来源折；海马共同来源队列共同留出、分别算AUC。填补、标准化与模型只拟合当前折训练数据。', '',
        '本次只在固定六基因内展开顺序，不重选六基因底座。全开发集确定的候选与基因保留不变，内部排序与规模搜索使用相同CV分数，属于探索性内部评价。', '',
        '已保存75个候选组合的试算结果、各步OOF预测、逐GEO AUC及输入校验。第六点与原预测逐供体核对，第七至六十点逐项保持一致。原图件和手稿未覆盖。']
    (OUT / '结果说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


if __name__ == '__main__':
    for name in ('结果', '图件', '核对'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        curve, summary, orders = compute()
        plot(curve, summary)
        report(curve, summary, orders)
    (OUT / '核对/运行警告.json').write_text(json.dumps(sorted({str(w.message) for w in caught}), ensure_ascii=False, indent=2), encoding='utf-8')
    print('DONE', OUT, flush=True)
