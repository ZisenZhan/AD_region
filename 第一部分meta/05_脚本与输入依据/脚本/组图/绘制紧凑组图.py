"""Compact mixed-modality figures using the current, audited source tables."""
from pathlib import Path
from functools import partial
import json
import sys
import textwrap

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

O = Path(__file__).resolve().parents[1]
D = O / '源数据'
R = ['cerebellum', 'occipital_visual', 'frontal_prefrontal', 'temporal_neocortex', 'hippocampus', 'entorhinal_cortex']
RN = ['Cerebellum', 'Occipital', 'Frontal', 'Temporal', 'Hippocampus', 'Entorhinal']
SHORT = ['CB', 'OCC', 'F', 'T', 'HIP', 'EC']
COL = dict(zip(R, ['#00A88F', '#4CB6E0', '#7A4BE1', '#EF3D94', '#F3AA32', '#F04D47']))
DOWN, UP, INK, NEUTRAL = '#5042BD', '#EF334F', '#202129', '#C9C9D1'
CM = LinearSegmentedColormap.from_list('signed', ['#3C28AC', '#A59AE1', '#FCFCFF', '#F4A0AA', '#EB2948'])
CM.set_bad('#E8E8EC')
ENR = LinearSegmentedColormap.from_list('enrichment', ['#FF222F', '#E90077', '#A000BC', '#4930D5'])
mpl.rcParams.update({'font.family': 'Arial', 'font.size': 6.5, 'axes.labelsize': 6.2,
    'axes.titlesize': 7.0, 'xtick.labelsize': 5.7, 'ytick.labelsize': 5.7,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .55,
    'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'legend.fontsize': 5.7, 'legend.frameon': False, 'svg.fonttype': 'none',
    'pdf.fonttype': 42, 'axes.unicode_minus': True, 'savefig.facecolor': 'white'})


def read(name):
    return pd.read_csv(D / name, float_precision='round_trip', low_memory=False)


M = {r: read(r + '_Meta复算.csv.gz').set_index('gene_symbol') for r in ['whole_brain', *R]}
roster = read('最新队列与样本覆盖.csv')
counts = read('共同背景显著性与重复性.csv.gz').set_index('gene_symbol')
emb = read('全部GEO样本级tSNE.csv.gz')
colinfo = read('热图列信息.csv')
top = read('Top50预定规则名单.csv')
heat = read('跨队列及区域Meta热图.csv')
ac = read('六机制cameraPR区域检验.csv')
go = read('GO_BP方向性结果.csv.gz')
slopes = read('分层变化汇总.csv').set_index('region')
spec = read('固定24功能过程.csv').sort_values('order')
cts = ['Excitatory', 'Inhibitory', 'Astrocyte', 'Microglia_PVM', 'Oligodendrocyte', 'OPC', 'Endothelial', 'Pericyte']
ctlabels = ['Exc.', 'Inh.', 'Astro.', 'Micro./PVM', 'Oligo.', 'OPC', 'Endo.', 'Pericyte']
domains = ['Synapse', 'Mitochondrial Metabolism', 'Immune Response', 'Structural Stabilization', 'Proteostasis', 'Lipid Metabolism']
domain_labels = ['Synapse', 'Mitochondrial\nmetabolism', 'Immune response', 'Structural\nstabilization', 'Proteostasis', 'Lipid metabolism']
fixed = [('GO:0099504', 'Synaptic vesicle cycle'), ('GO:0099003', 'Synaptic vesicle transport'),
    ('GO:0007269', 'Neurotransmitter secretion'), ('GO:0010975', 'Neuron projection development'),
    ('GO:0006119', 'Oxidative phosphorylation'), ('GO:0042775', 'Mitochondrial ATP synthesis'),
    ('GO:0032543', 'Mitochondrial translation'), ('GO:0050727', 'Inflammatory response regulation'),
    ('GO:0061900', 'Glial cell activation'), ('GO:0002218', 'Innate immune activation'),
    ('GO:1901342', 'Vasculature development'), ('GO:0030198', 'Extracellular matrix organization'),
    ('GO:0042060', 'Wound healing'), ('GO:0016236', 'Macroautophagy'),
    ('GO:0043161', 'Proteasomal protein catabolism'), ('GO:0006457', 'Protein folding'),
    ('GO:0006412', 'Translation'), ('GO:0006397', 'mRNA processing')]
MANIFEST, EXPORTS = [], []


def title(ax, value):
    ax.set_title(value, loc='left', fontweight='bold', pad=5)


def clean_heat(ax):
    ax.tick_params(which='both', length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)


def inset_cbar(ax, im, label, ticks=None, position=(1.025, .14, .035, .60)):
    ca = ax.inset_axes(position)
    cb = ax.figure.colorbar(im, cax=ca, ticks=ticks)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5.0, length=1.8, pad=1)
    cb.set_label(label, fontsize=5.3, labelpad=3)


def coverage(ax):
    x = roster.groupby('region')[['AD', 'control']].sum().reindex(R)
    y = np.arange(6)
    ax.barh(y, x.control, color='#CDC9D4', height=.48, label='Control')
    ax.barh(y, x.AD, left=x.control, color=UP, height=.48, label='AD')
    ax.set_yticks(y, RN)
    ax.invert_yaxis()
    ax.set_xlabel('Donor-tissue records')
    for i, n in enumerate(x.sum(axis=1)):
        ax.text(n + 10, i, str(n), va='center', fontsize=5.5)
    ax.set_xlim(0, 1000)
    ax.set_xticks([0, 400, 800])
    ax.legend(ncol=2, loc='upper right', bbox_to_anchor=(1.08, 1.22), handlelength=1.0, columnspacing=.8)
    title(ax, 'Regional coverage')


def tsne(ax, geo, heading=True):
    z = emb[emb.geo.eq(geo)]
    for r in R:
        for label in [0, 1]:
            v = z[z.region.eq(r) & z.y.eq(label)]
            ax.scatter(v.tsne1, v.tsne2, s=3.8 if len(z) > 150 else 6.5,
                facecolors=COL[r] if label else 'none', edgecolors=COL[r], lw=.38, alpha=.88)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['left', 'bottom']].set_visible(False)
    if heading:
        ax.set_title(geo, fontsize=6.2, loc='left', pad=3, fontweight='bold')
    ax.margins(.08)


def tsne_three(ax):
    ax.axis('off')
    for i, geo in enumerate(['GSE118553', 'GSE48350', 'GSE36980']):
        tsne(ax.inset_axes([i / 3 + .015, .13, .29, .72]), geo)
    hs = [Line2D([], [], marker='o', color=COL[r], ls='', ms=2.8, label=label) for r, label in zip(R, SHORT)]
    ax.legend(handles=hs, loc='lower center', ncol=6, bbox_to_anchor=(.48, -.10), fontsize=5.0,
        handletextpad=.2, columnspacing=.7)
    title(ax, 'Within-cohort sample structure')


def volcano(ax):
    z = M['whole_brain']
    y = -np.log10(z.fdr.clip(1e-300))
    x = z.effect.to_numpy()
    color = np.where(z.fdr.ge(.05), '#C5C4D0', np.where(z.effect > 0, UP, DOWN))
    ax.scatter(np.clip(x, -1.5, 1.5), np.minimum(y, 20), s=2.0, c=color, alpha=.58, lw=0, rasterized=True)
    clip = (abs(x) > 1.5) | (y > 20)
    ax.scatter(np.clip(x[clip], -1.5, 1.5), np.minimum(y[clip], 20), s=4.0, c=np.array(color)[clip],
        marker='x', lw=.45, rasterized=True)
    ax.axhline(-np.log10(.05), color='#92909F', lw=.5, ls='--')
    ax.axvline(0, color='#92909F', lw=.5)
    ax.set(xlim=(-1.57, 1.57), ylim=(-.3, 21.8), xticks=[-1, 0, 1], yticks=[0, 10, 20])
    ax.set_xlabel('Random-effects coefficient')
    ax.set_ylabel('-log10(BH FDR)')
    for g in ['SNAP25', 'SYP', 'CCL2']:
        if g in z.index and z.loc[g, 'fdr'] < .05:
            xx, yy = z.loc[g, 'effect'], min(20, -np.log10(z.loc[g, 'fdr']))
            ax.annotate(g, (xx, yy), (xx + (.2 if xx > 0 else -.20), yy + 1.25),
                ha='left' if xx > 0 else 'right', fontsize=5.2, color=UP if xx > 0 else DOWN,
                arrowprops=dict(arrowstyle='-', lw=.4, color='#868291'))
    title(ax, 'Whole-brain Meta')


def method_agreement(ax):
    z = read('Meta与SumRank直接数据.csv')
    ax.scatter(z.signed_meta_strength, z.sumrank_signed_strength, s=1.8, color=DOWN, alpha=.25, lw=0, rasterized=True)
    ax.axhline(0, color='#ACA9B6', lw=.4)
    ax.axvline(0, color='#ACA9B6', lw=.4)
    ax.set_xlabel('Meta signed -log10(FDR)')
    ax.set_ylabel('Signed SumRank')
    ax.text(.05, .93, r'$\rho$ = ' + f'{stats.spearmanr(z.signed_meta_strength,z.sumrank_signed_strength).statistic:.3f}',
        transform=ax.transAxes, va='top', fontsize=6.3)
    title(ax, 'Cross-method agreement')


def recurrence(ax):
    n = np.arange(1, 7)
    values = np.array([(counts.significant_regions >= k).sum() for k in n])
    ax.bar(n, values, color=[DOWN, '#6852CA', '#8852C3', '#AB46A8', '#D13B7A', UP], width=.48)
    ax.set_yscale('log')
    ax.set_ylim(1, 30000)
    ax.set_xticks(n)
    ax.set_xlabel('At least n significant regions')
    ax.set_ylabel('Genes')
    for k, v in zip(n, values):
        ax.text(k, v * 1.19, f'{v:,}', ha='center', fontsize=5.3)
    title(ax, 'Regional recurrence')


ORA = {}
for db in ['GO_BP', 'KEGG']:
    v = read(db + '_重复基因ORA复算.csv')
    ORA[db] = v[v.ora_fdr.lt(.05) & v.overlap_count.ge(3)].head(8)
qs = np.concatenate([z.ora_fdr.to_numpy() for z in ORA.values()])
ORA_NORM = LogNorm(vmin=qs.min(), vmax=qs.max())


def enrichment(ax, db):
    z = ORA[db].iloc[::-1]
    ax.barh(np.arange(len(z)), z.overlap_count, color=ENR(ORA_NORM(z.ora_fdr)), height=.48)
    labels = [textwrap.fill(t.replace(' - ', ': '), 27) for t in z.term_name]
    ax.set_yticks(np.arange(len(z)), labels)
    ax.tick_params(axis='y', length=0, labelsize=5.5, pad=3)
    ax.set_xlabel('Gene count')
    ax.grid(axis='x', color='#E9E7EE', lw=.4)
    ax.set_axisbelow(True)
    ax.margins(y=.055, x=.08)
    title(ax, 'GO biological processes' if db == 'GO_BP' else 'KEGG pathways')
    if db == 'KEGG':
        inset_cbar(ax, mpl.cm.ScalarMappable(norm=ORA_NORM, cmap=ENR), 'BH FDR',
            ticks=[qs.min(), np.sqrt(qs.min()*qs.max()), qs.max()], position=(1.025, .15, .035, .62))
        ax.child_axes[-1].yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.0e'))


def cohort_heat(ax):
    h = heat.pivot(index='gene_symbol', columns='column', values='effect').reindex(index=top.gene_symbol, columns=colinfo.column)
    cohort_columns = colinfo.loc[~colinfo.is_meta, 'column'].tolist()
    ranks = h[cohort_columns].rank(axis=0, method='average', pct=True)

    def shared_distance(a, b):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 2:
            raise ValueError('Insufficient shared observations for clustering; missing values are not imputed.')
        return np.sqrt(np.mean((a[ok] - b[ok]) ** 2))

    row_tree = linkage(pdist(ranks.to_numpy(), metric=shared_distance), method='average', optimal_ordering=True)
    gene_order = h.index[leaves_list(row_tree)]
    columns, trees = [], {}
    for r in R:
        group = colinfo[colinfo.region.eq(r)]
        cohort_ids = group.loc[~group.is_meta, 'column'].tolist()
        tree = linkage(pdist(ranks[cohort_ids].to_numpy().T, metric=shared_distance),
            method='average', optimal_ordering=True)
        trees[r] = tree
        columns.extend([cohort_ids[i] for i in leaves_list(tree)])
        columns.extend(group.loc[group.is_meta, 'column'].tolist())
    display_info = colinfo.set_index('column').loc[columns].reset_index()
    display_info['display_column'] = np.arange(len(display_info))
    assert len(columns) == len(set(columns)) == 34
    for r in R:
        assert display_info.loc[display_info.region.eq(r)].iloc[-1].is_meta
    h = h.loc[gene_order, columns]
    pd.DataFrame({'gene_symbol': gene_order, 'display_row': np.arange(len(gene_order))}).to_csv(
        D / '图2A_基因行聚类顺序.csv', index=False, encoding='utf-8-sig')
    display_info.to_csv(D / '图2A_脑区内队列聚类与Meta固定列.csv', index=False, encoding='utf-8-sig')
    h.rename_axis('gene_symbol').reset_index().to_csv(D / '图2A_双向聚类原始系数.csv',
        index=False, encoding='utf-8-sig')
    pd.DataFrame(row_tree, columns=['left_node','right_node','distance','n_leaves']).to_csv(
        D / '图2A_基因行层次聚类连接矩阵.csv', index=False, encoding='utf-8-sig')
    (D / '图2A_聚类设置.json').write_text(json.dumps(dict(
        row_features='28 individual cohort columns only; six Meta columns excluded',
        column_groups='cluster within each region; keep region blocks fixed and Meta at each block end',
        transformation_for_distance='within-column percentile ranks among the same 50 displayed genes',
        metric='RMS Euclidean difference on shared finite observations; at least two observations',
        linkage='average', optimal_ordering=True, missing_imputation=False,
        heatmap_values='unchanged raw coefficients; rank transform only affects display ordering'), indent=2), encoding='utf-8')
    im = ax.imshow(np.ma.masked_invalid(h), cmap=CM, vmin=-.8, vmax=.8, aspect='auto', interpolation='nearest')
    ax.set_yticks(np.arange(50), gene_order, fontsize=5.3)
    ax.set_xticks(np.arange(len(display_info)), [s.replace('GSE', '') if s != 'Meta' else s for s in display_info.label], rotation=90, fontsize=5.0)
    for r, label in zip(R, RN):
        ii = display_info.loc[display_info.region.eq(r), 'display_column'].to_numpy()
        transform = ax.get_xaxis_transform()
        ax.plot([ii[0]-.5, ii[-1]+.5], [1.055, 1.055], color=COL[r], lw=2.5,
            transform=transform, clip_on=False, solid_capstyle='butt')
        ax.text(ii.mean(), 1.075, label, ha='center', va='bottom', rotation=42,
            transform=transform, fontsize=5.6, clip_on=False)
        ax.axvline(ii[-1] + .5, color='white', lw=.9)
        left, width = ii[0] / len(display_info), (len(ii)-1) / len(display_info)
        da = ax.inset_axes([left, 1.002, width, .042])
        dendrogram(trees[r], ax=da, orientation='top', no_labels=True, color_threshold=0,
            above_threshold_color='#948B9F')
        for line in da.collections:
            line.set_linewidth(.40)
        da.axis('off')
    for row in display_info[display_info.is_meta].itertuples():
        ax.add_patch(Rectangle((row.display_column-.5, -.5), 1, 50, fill=False, ec='#202129', lw=.5))
    q = heat.pivot(index='gene_symbol', columns='column', values='fdr').loc[gene_order, columns]
    significant = q.lt(.05).to_numpy()
    yy, xx = np.nonzero(significant)
    ax.plot(xx, yy, linestyle='None', marker='o', ms=.85, mec='none',
        color='#17121E', alpha=.95, zorder=4)
    marker_data = heat.copy()
    marker_data['display_row'] = marker_data.gene_symbol.map(dict(zip(gene_order, range(50))))
    marker_data['display_column'] = marker_data.column.map(dict(zip(columns, range(34))))
    marker_data['significance_dot'] = marker_data.fdr.lt(.05)
    marker_data.sort_values(['display_row', 'display_column']).to_csv(
        D / '图2A_全部队列及Meta显著性黑点.csv', index=False, encoding='utf-8-sig')
    axis_width_in = ax.get_position().width * ax.figure.get_figwidth()
    da = ax.inset_axes([-.65 / axis_width_in, 0, .14 / axis_width_in, 1])
    dendrogram(row_tree, ax=da, orientation='left', no_labels=True, color_threshold=0,
        above_threshold_color='#948B9F')
    da.set_ylim(500, 0)
    for line in da.collections:
        line.set_linewidth(.40)
    da.axis('off')
    clean_heat(ax)
    ax.set_ylim(49.5, -.5)
    inset_cbar(ax, im, 'Meta / GEO coefficient', [-.8, 0, .8], position=(.15, -.10, .65, .014))
    ca = ax.child_axes[-1]
    ca.clear()
    cb = ax.figure.colorbar(im, cax=ca, orientation='horizontal', ticks=[-.8, 0, .8])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5, length=1.5)
    cb.set_label('Meta / GEO coefficient', fontsize=5.8, labelpad=1.2)


def domain_heat(ax, sensitivity=False, full_names=False):
    field = 'sensitivity_FDR_005' if sensitivity else 'FDR'
    z = ac[ac.scope.isin(R)].copy()
    z['value'] = np.where(z.Direction.eq('Up'), 1, -1) * -np.log10(z[field].clip(1e-30))
    h = z.pivot(index='mechanism', columns='scope', values='value').reindex(index=domains, columns=R)
    im = ax.imshow(h, cmap=CM, vmin=-5, vmax=5, aspect='auto')
    ax.set_yticks(np.arange(6), domain_labels, fontsize=5.5)
    ax.set_xticks(np.arange(6), RN if full_names else SHORT,
        rotation=55 if full_names else 0, ha='right' if full_names else 'center', fontsize=5.2)
    for i, a in enumerate(domains):
        for j, r in enumerate(R):
            q = z[z.mechanism.eq(a)&z.scope.eq(r)][field].iloc[0]
            if q < .05:
                ax.text(j, i, '*', va='center', ha='center', fontsize=6.3,
                    color='white' if abs(h.iloc[i,j])>2.8 else INK)
    clean_heat(ax)
    title(ax, 'Six biological domains' if not sensitivity else 'Correlation assumption 0.05')
    inset_cbar(ax, im, 'Signed -log10(FDR)', [-5, 0, 5])


def go_heat(ax):
    z = go[go.scope.isin(R)]
    pool = dict(fixed)
    pool.update(dict(zip(spec.set_id, spec.label)))
    present = [i for i in pool if i in set(z.go_id)]
    h = z.pivot(index='go_id', columns='scope', values='directional_rank_z').reindex(index=present, columns=R)
    assert h.notna().all().all(), 'Clustering requires complete profiles; do not impute missing tests.'
    standardized = (h.to_numpy() - h.to_numpy().mean(axis=1, keepdims=True)) / h.to_numpy().std(axis=1, keepdims=True)
    assert np.isfinite(standardized).all()
    tree = linkage(standardized, method='average', metric='euclidean', optimal_ordering=True)
    h = h.iloc[leaves_list(tree)]
    aliases = {'GO:0010975': 'Neuron projection development',
        'GO:0042775': 'Mitochondrial ATP synthesis',
        'GO:0050727': 'Inflammatory regulation',
        'GO:0030198': 'Extracellular matrix organization',
        'GO:0030968': 'ER unfolded protein response'}
    ordered = pd.DataFrame({'go_id': h.index, 'display_label': [aliases.get(i, pool[i]) for i in h.index],
        'row_order': np.arange(len(h)), 'tested_regions': 6})
    ordered.to_csv(D / '图2C_GO聚类顺序与标签.csv', index=False, encoding='utf-8-sig')
    h.rename_axis('go_id').reset_index().to_csv(D / '图2C_聚类热图原始Z.csv', index=False, encoding='utf-8-sig')
    audit = pd.DataFrame({'go_id': list(pool), 'label': list(pool.values()),
        'displayed': [i in present for i in pool],
        'reason': ['eligible in all six regions' if i in present else 'not eligible under original measurable-gene size filter' for i in pool]})
    audit.to_csv(D / '图2C_37过程显示资格.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(tree, columns=['left_node', 'right_node', 'distance', 'n_leaves']).to_csv(
        D / '图2C_行层次聚类连接矩阵.csv', index=False, encoding='utf-8-sig')
    im = ax.imshow(np.ma.masked_invalid(h), cmap=CM, vmin=-12, vmax=12, aspect='auto')
    ax.set_yticks(np.arange(len(h)), ordered.display_label, fontsize=5.1)
    ax.set_xticks(np.arange(6), RN, rotation=55, ha='right', fontsize=5.2)
    for i, term in enumerate(h.index):
        for j, r in enumerate(R):
            v = z[z.go_id.eq(term)&z.scope.eq(r)]
            if v.empty:
                ax.text(j, i, 'x', ha='center', va='center', fontsize=5.0, color='#898592')
            elif v.directional_rank_fdr.iloc[0] < .05:
                ax.text(j, i, '*', ha='center', va='center', fontsize=5.5,
                    color='white' if abs(h.iloc[i,j])>7 else INK)
    clean_heat(ax)
    axis_width_in = ax.get_position().width * ax.figure.get_figwidth()
    da = ax.inset_axes([-1.35 / axis_width_in, 0, .16 / axis_width_in, 1])
    dendrogram(tree, ax=da, orientation='left', no_labels=True, color_threshold=0,
        above_threshold_color='#948B9F')
    da.set_ylim(10 * len(h), 0)
    for line in da.collections:
        line.set_linewidth(.45)
    da.axis('off')
    title(ax, 'Clustered functional processes')
    inset_cbar(ax, im, 'Directional rank Z', [-12, 0, 12])


def slope(ax, kind):
    for r in R:
        values = slopes.loc[r, [kind+'_whole', kind+'_region']].to_numpy(dtype=float)
        ax.plot([0,1], values, '-o', color=COL[r], lw=1.1, ms=3.7, mec='white', mew=.45)
    ax.set_xticks([0,1], ['Whole brain','Regional'])
    ax.set_xlim(-.14,1.14)
    ax.grid(axis='y',color='#E7E4EE',lw=.45)
    ax.set_axisbelow(True)
    ax.set_ylabel('Median |model-derived g*|' if kind=='abs_g' else 'Median I-squared (%)')
    title(ax, 'Effect magnitude' if kind=='abs_g' else 'Between-study heterogeneity')


def template_heat(ax):
    z = read('额叶全部细胞全部过程.csv')
    h = z.pivot(index='set_id', columns='cell_type', values='template_p').reindex(index=spec.set_id, columns=cts)
    directions = z.pivot(index='set_id', columns='cell_type', values='template_direction').reindex_like(h)
    q = z.pivot(index='set_id', columns='cell_type', values='template_q').reindex_like(h)
    a = -np.log10(h.clip(1e-20)) * np.where(directions.eq('Up'),1,-1)
    im = ax.imshow(np.ma.masked_invalid(a), cmap=CM, vmin=-4, vmax=4, aspect='auto')
    ax.set_yticks(np.arange(24), spec.label, fontsize=5.4)
    ax.set_xticks(np.arange(8), ctlabels, rotation=60, ha='right', fontsize=5.2)
    for i in range(24):
        for j in range(8):
            if np.isfinite(q.iloc[i,j]) and q.iloc[i,j] < .05:
                ax.plot(j,i,'o',ms=1.9,mec='none',color='#17121E')
    ax.add_patch(Rectangle((1.5,-.5),1,24,fill=False,ec='#16101F',lw=.8))
    order=spec.functional_group.tolist()
    for i in range(1,24):
        if order[i] != order[i-1]:
            ax.axhline(i-.5,color='white',lw=.85)
    clean_heat(ax)
    title(ax, 'Frontal bulk-pattern correspondence')
    inset_cbar(ax, im, 'Signed -log10(P)', [-4,0,4])


def venn(ax):
    x = read('三来源交集分区.csv')
    counts3 = dict(zip(x.membership.astype(str).str.zfill(3), x.genes))
    ax.set(aspect='equal', xlim=(-1.9,1.9), ylim=(-1.85,1.65))
    ax.axis('off')
    for (xc,yc),col in zip([(-.57,.24),(.57,.24),(0,-.58)],['#50A5EF','#FA6978','#8C6DE0']):
        ax.add_patch(Circle((xc,yc),1.04,facecolor=mpl.colors.to_rgba(col,.35),ec='white',lw=.6))
    locations=[('100',-1.03,.35),('010',1.03,.35),('001',0,-1.13),('110',0,.80),('101',-.59,-.43),('011',.59,-.43),('111',0,.03)]
    for key,xp,yp in locations:
        ax.text(xp,yp,f'{counts3.get(key,0):,}',ha='center',va='center',fontsize=7.5 if key=='111' else 6.7,fontweight='bold' if key=='111' else 'normal')
    for xp,yp,label in [(-.93,1.48,'Frontal bulk'),(.95,1.48,'SEA-AD'),(0,-1.78,'ROSMAP-427')]:
        ax.text(xp,yp,label,ha='center',va='center',fontsize=6.5,fontweight='bold')
    title(ax,'Genome-wide significant genes')


def heat67(ax):
    cols=['bulk_beta','sea_beta','ros_beta']
    z=read('三来源67同向基因.csv').sort_values('gene_symbol').set_index('gene_symbol')
    z['mean_beta']=z[cols].mean(axis=1)
    z=z.sort_values('mean_beta',ascending=False,kind='stable')
    assert len(z)==67 and z.mean_beta.is_monotonic_decreasing
    z.reset_index().to_csv(D/'图3C_67基因原始效应及算术均值.csv',index=False,encoding='utf-8-sig')
    raw=z[cols+['mean_beta']].to_numpy()
    scaled=raw/np.abs(raw).max(axis=0)
    im=ax.imshow(scaled,cmap=CM,vmin=-1,vmax=1,aspect='auto',interpolation='nearest')
    ax.set_yticks(np.arange(len(z)),z.index,fontsize=5.2)
    ax.set_xticks(range(4),['Bulk','SEA-AD','ROSMAP','Mean'],fontsize=5.8)
    ax.xaxis.tick_top()
    for i,j in np.ndindex(raw.shape):
        ax.text(j,i,f'{raw[i,j]:+.2f}',ha='center',va='center',fontsize=5.0,
                color='white' if abs(scaled[i,j])>.68 else INK)
    ax.axvline(2.5,color='#8F8599',lw=.65)
    ax.set_xticks(np.arange(5)-.5,minor=True)
    ax.set_yticks(np.arange(len(z)+1)-.5,minor=True)
    ax.grid(which='minor',color='white',lw=.30)
    clean_heat(ax)
    ca=ax.inset_axes([.10,-.032,.80,.01])
    cb=ax.figure.colorbar(im,cax=ca,orientation='horizontal',ticks=[-1,0,1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5,length=1.6)
    cb.set_label('Column-scaled effect',fontsize=5.6,labelpad=2)


def paired_i2(ax):
    p=read('逐基因配对分层变化.csv.gz')
    boxes=[p.loc[p.region.eq(r),'I2_change'].to_numpy() for r in R]
    art=ax.boxplot(boxes,widths=.52,showfliers=False,patch_artist=True,
        medianprops=dict(color=INK,lw=.8),boxprops=dict(lw=.5),whiskerprops=dict(lw=.5),capprops=dict(lw=.5))
    for patch,r in zip(art['boxes'],R):patch.set_facecolor(COL[r]);patch.set_alpha(.40)
    ax.axhline(0,color='#9F99AB',ls='--',lw=.5)
    ax.set_xticks(np.arange(1,7),SHORT)
    ax.set_ylabel('Regional - whole-brain I-squared (pp)')
    title(ax,'Paired gene-wise changes')


def global_scatter(ax, source):
    z=read('三来源共同背景与全部统计.csv.gz')
    x=stats.rankdata(z.bulk_beta)/len(z)
    y=stats.rankdata(z[source+'_beta'])/len(z)
    ax.scatter(x,y,s=1.2,alpha=.18,lw=0,color=DOWN,rasterized=True)
    ax.set(xlim=(-.025,1.025),ylim=(-.025,1.025),xticks=[0,.5,1],yticks=[0,.5,1])
    ax.set_xlabel('Frontal bulk effect percentile')
    ax.set_ylabel(('SEA-AD' if source=='sea' else 'ROSMAP')+' effect percentile')
    rho=stats.spearmanr(x,y).statistic
    ax.text(.04,.94,r'$\rho$ = '+f'{rho:.3f}',transform=ax.transAxes,va='top',fontsize=6.3)
    title(ax,'All shared testable genes')


def newfig(size):
    return plt.figure(figsize=size,facecolor='white')

def new_go(ax):
    from 六领域热图 import plot
    plot(ax,go,R,RN,cts,ctlabels,CM,'bulk')

def new_template(ax):
    from 六领域热图 import plot
    plot(ax,go,R,RN,cts,ctlabels,CM,'template')

def domain_cells(ax):
    from 六领域热图 import AX, COL
    tab=pd.read_csv(O/'六领域GO与额叶单核/全量六机制总览/六机制八类细胞检验.csv')
    def mat(col):
        return tab.pivot(index='go_id',columns='cell_type',values=col).loc[AX,cts]
    value=-np.log10(mat('p').clip(lower=1e-20))*np.where(mat('direction')=='Up',1,-1)
    im=ax.imshow(value,cmap=CM,vmin=-4,vmax=4,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(8),ctlabels,rotation=60,ha='right',fontsize=5.1)
    ax.set_yticks(range(6),['Synapse','Mitochondrial metabolism','Immune response','Structural stabilization','Proteostasis','Lipid metabolism'],fontsize=5.5)
    clean_heat(ax)
    pts=ax.get_position().width*ax.figure.get_figwidth()*72
    ax.tick_params(axis='y',pad=8)
    for i,color in enumerate(COL):
        ax.add_patch(Rectangle((-6.3/pts,i-.5),4.8/pts,1,transform=ax.get_yaxis_transform(),
            fc=color,ec='none',clip_on=False))
    for i,j in zip(*np.where(mat('fdr')<.05)):
        ax.plot(j,i,'o',color='white' if abs(value.iloc[i,j])>2.8 else INK,ms=1.9,mec='none')
    ax.add_patch(Rectangle((1.5,-.5),1,6,fill=False,ec=INK,lw=.8))
    title(ax,'Six biological domains')
    inset_cbar(ax,im,'Signed -log10(P)',[-4,0,4],position=(1.04,.18,.037,.60))


def add(fig, rect, draw, label, standalone_size=(3.6,2.7)):
    ax=fig.add_axes(rect)
    draw(ax)
    return dict(ax=ax,draw=draw,label=label,size=standalone_size)


def export(fig,name,folder,panels,letter_positions=None):
    skip = ('--figure2' in sys.argv and not name.startswith('图02_'))
    skip |= ('--figure3' in sys.argv and not name.startswith('图03_'))
    skip |= ('--figures2-3' in sys.argv and not name.startswith(('图02_', '图03_')))
    if skip:
        MANIFEST.append(dict(figure=name, folder=folder, panels=len(panels)))
        plt.close(fig)
        return
    for i,p in enumerate(panels):
        if letter_positions:
            x,y=letter_positions[i]
        else:
            box=p['ax'].get_position();x=max(.007,box.x0-.06);y=min(.985,box.y1+.025)
        fig.text(x,y,chr(65+i),fontsize=11,fontweight='normal',ha='left',va='top')
    fig.canvas.draw()
    for ext in ['png','pdf','svg']:
        fig.savefig(O/folder/(name+'.'+ext),dpi=420,facecolor='white')
    plt.close(fig)
    for i,p in enumerate(panels):
        f=newfig(p['size'])
        a=f.add_axes([.28,.21,.62,.65])
        if p['draw'] in [tsne_three,heat67,venn]:a.set_position([.08,.12,.84,.77])
        if p['draw'] is heat67:a.set_position([.25,.09,.69,.85])
        p['draw'](a)
        for ext in ['png','pdf','svg']:
            f.savefig(O/'单独子图'/(name+'_'+chr(65+i)+'_'+p['label']+'.'+ext),dpi=360,bbox_inches='tight',pad_inches=.07)
        plt.close(f)
    MANIFEST.append(dict(figure=name,folder=folder,panels=len(panels)))
    print(name,flush=True)


def main():
    f=newfig((7.205,7.35))
    p=[add(f,[.14,.77,.29,.175],coverage,'样本覆盖'),
       add(f,[.51,.765,.46,.19],tsne_three,'三队列样本结构',(6.2,2.5)),
       add(f,[.085,.47,.235,.205],volcano,'整脑火山图'),
       add(f,[.415,.47,.235,.205],method_agreement,'Meta与SumRank对应'),
       add(f,[.755,.47,.205,.205],recurrence,'跨脑区重复性'),
       add(f,[.285,.085,.185,.25],partial(enrichment,db='GO_BP'),'重复基因GO',(4.0,3.5)),
       add(f,[.775,.085,.14,.25],partial(enrichment,db='KEGG'),'重复基因KEGG',(4.0,3.5))]
    export(f,'图01_共同转录异常与跨脑区重复','主文组图',p,[(.015,.975),(.48,.975),(.015,.71),(.345,.71),(.68,.71),(.015,.37),(.51,.37)])
    f=newfig((7.205,7.65))
    p=[add(f,[.75,.785,.195,.14],partial(domain_heat,full_names=True),'六机制区域模式',(4.3,2.9)),
       add(f,[.848,.278,.098,.355],new_go,'具体GO区域模式',(5.0,5.8)),
       add(f,[.11,.06,.29,.115],partial(slope,kind='abs_g'),'效应量变化'),
       add(f,[.61,.06,.29,.115],partial(slope,kind='I2'),'异质性变化')]
    hs=[Line2D([],[],color=COL[r],marker='o',ms=2.8,lw=1,label=label) for r,label in zip(R,RN)]
    f.legend(handles=hs,loc='lower center',bbox_to_anchor=(.51,.008),ncol=6,fontsize=5.5,handlelength=1.2,columnspacing=1)
    if '--figure3' not in sys.argv:
        from 嵌入新版图2A import export_current
        export_current(f,p,[(.015,.98),(.61,.98),(.61,.705),(.015,.215),(.51,.215)],O)
        solo=newfig((5,5.8))
        new_go(solo.add_axes([.54,.14,.31,.76]))
        for ext in ['png','svg','pdf']:
            solo.savefig(O/'单独子图'/('图02_区域机制模式与分层收益_C_具体GO区域模式.'+ext),dpi=360,bbox_inches='tight',pad_inches=.07)
        plt.close(solo)
    plt.close(f)
    MANIFEST.append(dict(figure='图02_区域机制模式与分层收益',folder='主文组图',panels=5))
    f=newfig((7.205,9.05))
    p=[add(f,[.30,.833,.23,.110],domain_cells,'全量GO六机制与细胞呼应',(4.5,2.6)),
       add(f,[.30,.357,.23,.365],new_template,'额叶显著GO与细胞方向呼应',(5.3,5.1)),
       add(f,[.12,.023,.37,.247],venn,'三来源显著基因交集',(4.0,3.5)),
       add(f,[.704,.084,.275,.865],heat67,'67同向基因三来源效应',(3.2,7.9))]
    export(f,'图03_额叶星形胶质细胞的跨来源呼应','主文组图',p,[(.015,.98),(.015,.774),(.015,.288),(.605,.98)])
    f=newfig((7.205,8.05));p=[]
    for i,geo in enumerate(sorted(emb.geo.unique())):
        p.append(add(f,[.065+(i%4)*.24,.825-(i//4)*.179,.19,.125],partial(tsne,geo=geo),geo,(3.2,2.6)))
    export(f,'补充图01_全部队列样本结构','补充组图',p)
    pd.DataFrame(MANIFEST).to_csv(O/'核对/组图清单.csv',index=False,encoding='utf-8-sig')


if __name__=='__main__':
    main()
