"""Recompute summaries from the current cohort inventory and saved model outputs."""
from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
from scipy import stats

O = Path(__file__).resolve().parents[1]
B = O.parent
S = B / '第一部分_最新队列六机制重算_20261113'
P = B / '第一部分_最新六机制_论文组图_20261113'
E = B / '脑区功能失调模式与单核呼应_20261113'
F = B / '额叶分细胞分机制组图_20261115'
ROS = B / 'ROSMAP427_前额叶星形胶质_192基因验证'
R = ['cerebellum', 'occipital_visual', 'frontal_prefrontal', 'temporal_neocortex', 'hippocampus', 'entorhinal_cortex']
INPUTS = set()
for name in ['源数据', '核对', '主文组图', '补充组图', '单独子图', '说明']:
    (O / name).mkdir(parents=True, exist_ok=True)


def read(path, **kwargs):
    INPUTS.add(path)
    kwargs.setdefault('float_precision', 'round_trip')
    return pd.read_csv(path, **kwargs)


def save(df, name):
    df.to_csv(O / '源数据' / name, index=False, encoding='utf-8-sig', float_format='%.17g')


def bh(values):
    return stats.false_discovery_control(np.asarray(values, dtype=float))


def integrate(long):
    v = long.assign(se2=long.se ** 2).groupby(['family', 'gene_symbol'], sort=True).agg(
        logFC=('logFC', 'mean'), se2=('se2', 'mean'), g=('g', 'mean')).reset_index()
    v['se'] = np.sqrt(v.se2)
    y = v.pivot(index='gene_symbol', columns='family', values='logFC')
    se = v.pivot(index='gene_symbol', columns='family', values='se').reindex_like(y).to_numpy()
    gs = v.pivot(index='gene_symbol', columns='family', values='g').reindex_like(y)
    a = y.to_numpy()
    ok = np.isfinite(a) & np.isfinite(se) & (se > 0)
    n = ok.sum(axis=1)
    w = np.where(ok, 1 / se ** 2, 0)
    yy = np.where(ok, a, 0)
    fixed = (w * yy).sum(axis=1) / w.sum(axis=1)
    q = (w * np.where(ok, (a - fixed[:, None]) ** 2, 0)).sum(axis=1)
    c = w.sum(axis=1) - (w * w).sum(axis=1) / w.sum(axis=1)
    tau = np.maximum(0, np.divide(q - (n - 1), c, out=np.zeros_like(q), where=c > 0))
    wr = np.where(ok, 1 / (se ** 2 + tau[:, None]), 0)
    effect = (wr * yy).sum(axis=1) / wr.sum(axis=1)
    ese = np.sqrt(1 / wr.sum(axis=1))
    p = 2 * stats.norm.sf(abs(effect / ese))
    i2 = np.maximum(0, np.divide(q - (n - 1), q, out=np.zeros_like(q), where=q > 0)) * 100
    i2[n <= 1] = 0
    return pd.DataFrame(dict(gene_symbol=y.index, k=n, effect=effect, se=ese, p=p, fdr=bh(p),
        ci_low=effect - 1.96 * ese, ci_high=effect + 1.96 * ese, I2=i2,
        tau2=tau, median_g_star=gs.median(axis=1).to_numpy()))


def main():
    long = read(S / '输入/规范基因队列差异与模型衍生g.csv.gz')
    roster = read(S / '输入/最新全发现队列清单.csv')
    assert roster.geo.nunique() == 20 and len(roster) == 28 and roster.family.nunique() == 17
    assert roster.groupby('region').size().to_dict() == dict(cerebellum=2, occipital_visual=3,
        frontal_prefrontal=10, temporal_neocortex=5, hippocampus=6, entorhinal_cortex=2)
    save(roster, '最新队列与样本覆盖.csv')
    meta, checks = {}, []
    for scope in ['whole_brain', *R]:
        m = integrate(long if scope == 'whole_brain' else long[long.region.eq(scope)])
        old = read(S / '结果' / (scope + '_基因Meta.csv.gz')).set_index('gene_symbol')
        new = m.set_index('gene_symbol').loc[old.index]
        err = float(np.max(np.abs(new.effect - old.effect)))
        assert err < 1e-9 and np.max(abs(new.fdr - old.fdr)) < 1e-9, (scope, err)
        meta[scope] = m.set_index('gene_symbol')
        save(m, scope + '_Meta复算.csv.gz')
        checks.append(dict(scope=scope, n_genes=len(m), effect_max_abs_error=err,
            significant=int((m.fdr < .05).sum()), up=int(((m.fdr < .05) & (m.effect > 0)).sum()),
            down=int(((m.fdr < .05) & (m.effect < 0)).sum())))
    save(pd.DataFrame(checks), 'Meta复算及显著数量.csv')
    common = sorted(set.intersection(*(set(meta[r].index) for r in R)))
    flags = pd.DataFrame({r: meta[r].fdr.reindex(common).lt(.05) for r in R})
    flags['significant_regions'] = flags.sum(axis=1)
    save(flags.rename_axis('gene_symbol').reset_index(), '共同背景显著性与重复性.csv.gz')
    selected = flags[flags.significant_regions >= 3].index
    assert len(selected) == 1327
    ef = pd.DataFrame({r: meta[r].effect.reindex(common) for r in R})
    top = flags[flags.significant_regions >= 4].copy()
    top['ranking_mean_abs_meta_effect'] = ef.abs().where(flags[R]).mean(axis=1)
    top = top.reset_index().sort_values(['ranking_mean_abs_meta_effect', 'gene_symbol'], ascending=[False, True]).head(50)
    save(top, 'Top50预定规则名单.csv')
    info, rows = [], []
    for r in R:
        units = roster.loc[roster.region.eq(r)].sort_values('geo').unit.tolist()
        for unit in units + [r]:
            is_meta = unit == r
            z = meta[r] if is_meta else long[long.unit.eq(unit)].set_index('gene_symbol')
            effect = z.effect if is_meta else z.logFC
            q = z.fdr if is_meta else pd.Series(bh(z.p_value), index=z.index)
            col = len(info)
            info.append(dict(column=col, unit=unit, region=r, label='Meta' if is_meta else unit.split('__')[1], is_meta=is_meta))
            rows.extend(dict(gene_symbol=g, column=col, region=r, unit=unit, is_meta=is_meta,
                effect=effect.get(g, np.nan), fdr=q.get(g, np.nan)) for g in top.gene_symbol)
    save(pd.DataFrame(info), '热图列信息.csv')
    save(pd.DataFrame(rows), '跨队列及区域Meta热图.csv')
    paired, slopes = [], []
    for r in R:
        z = meta[r].join(meta['whole_brain'], how='inner', lsuffix='_region', rsuffix='_whole')
        z = z[(z.k_region >= 2) & (z.k_whole >= 2)].copy()
        z['I2_change'] = z.I2_region - z.I2_whole
        z['abs_g_change'] = abs(z.median_g_star_region) - abs(z.median_g_star_whole)
        paired.append(z.assign(region=r).reset_index())
        slopes.append(dict(region=r, n_genes=len(z), I2_whole=z.I2_whole.median(), I2_region=z.I2_region.median(),
            abs_g_whole=abs(z.median_g_star_whole).median(), abs_g_region=abs(z.median_g_star_region).median(),
            median_paired_I2_change=z.I2_change.median(), median_paired_abs_g_change=z.abs_g_change.median()))
    save(pd.concat(paired), '逐基因配对分层变化.csv.gz')
    save(pd.DataFrame(slopes), '分层变化汇总.csv')
    for filename in ['whole_brain_SumRank.csv.gz', '六机制cameraPR区域检验.csv', 'GO_BP方向性结果.csv.gz',
                     '标准化g尺度I2敏感性汇总.csv', '全部GEO样本级tSNE.csv.gz', 'tSNE参数与人数.csv']:
        z = read(P / '源数据' / filename)
        save(z, filename)
    sr = read(P / '源数据/whole_brain_SumRank.csv.gz').set_index('gene_symbol')
    zz = meta['whole_brain'].join(sr[['sumrank_signed_strength']], how='inner')
    zz['signed_meta_strength'] = np.sign(zz.effect) * -np.log10(zz.fdr.clip(1e-300))
    rho = stats.spearmanr(zz.signed_meta_strength, zz.sumrank_signed_strength)
    save(zz.reset_index(), 'Meta与SumRank直接数据.csv')
    # Recalculate every ORA P and BH value, including zero-overlap terms.
    for db in ['GO_BP', 'KEGG']:
        z = read(S / '结果' / (db + '_共同基因ORA.csv'))
        for genes in z.overlap_genes.fillna(''):
            assert set(filter(None, genes.split(';'))) <= set(selected)
        pv = stats.hypergeom.sf(z.overlap_count - 1, z.annotated_universe_size,
            z.term_size, z.selected_annotated_size)
        assert np.max(abs(pv - z.ora_p)) < 1e-10
        q = bh(pv)
        assert np.max(abs(q - z.ora_fdr)) < 1e-10
        z['ora_p'], z['ora_fdr'] = pv, q
        save(z.sort_values(['ora_fdr', 'term_id']), db + '_重复基因ORA复算.csv')
    astro = read(E / '单核结果/SEAAD__Astrocyte__local_CPS__frontal_prefrontal__regional__genes.csv.gz').set_index('gene_symbol')
    ros = read(ROS / 'inputs/Ast_gpath.csv', keep_default_na=False).set_index('gene')
    for key in ['logFC', 'p_val', 'p_adj.loc']:
        ros[key] = pd.to_numeric(ros[key], errors='coerce')
    a = meta['frontal_prefrontal'][['effect', 'fdr']].set_axis(['bulk_beta', 'bulk_q'], axis=1)
    b = astro[['logFC', 'adj.P.Val']].set_axis(['sea_beta', 'sea_q'], axis=1)
    c = ros[['logFC', 'p_adj.loc']].set_axis(['ros_beta', 'ros_q'], axis=1)
    j = a.join(b, how='inner').join(c, how='inner').dropna()
    flags = j[['bulk_q', 'sea_q', 'ros_q']].lt(.05)
    j['membership'] = flags.astype(int).astype(str).agg(''.join, axis=1)
    core = j[j.membership.eq('111')].copy()
    positive = core[['bulk_beta', 'sea_beta', 'ros_beta']].gt(0).all(axis=1)
    negative = core[['bulk_beta', 'sea_beta', 'ros_beta']].lt(0).all(axis=1)
    core['direction'] = np.select([positive, negative], ['Up', 'Down'], default='Mixed')
    agreed = core[core.direction.ne('Mixed')].copy()
    assert len(j) == 13416 and len(core) == 72 and len(agreed) == 67
    assert agreed.direction.value_counts().to_dict() == {'Up': 56, 'Down': 11}
    save(j.rename_axis('gene_symbol').reset_index(), '三来源共同背景与全部统计.csv.gz')
    save(core.rename_axis('gene_symbol').reset_index(), '三来源共同显著_含方向冲突.csv')
    save(agreed.rename_axis('gene_symbol').reset_index(), '三来源67同向基因.csv')
    save(j.membership.value_counts().rename_axis('membership').reset_index(name='genes'), '三来源交集分区.csv')
    correlations = []
    for col in ['sea_beta', 'ros_beta']:
        result = stats.spearmanr(j.bulk_beta, j[col])
        correlations.append(dict(comparison=col, n=len(j), rho=result.statistic, p=result.pvalue))
    save(pd.DataFrame(correlations), '全共同背景效应秩相关.csv')
    for filename in ['固定24功能过程.csv', '额叶Bulk全部功能过程.csv', '额叶全部细胞全部过程.csv', '额叶有效供体规模.csv']:
        save(read(F / '源数据' / filename), filename)
    save(read(ROS / '67基因三来源效应热图/上下调_GO_BP富集.csv.gz'), '67同向基因_GO_BP完整富集.csv.gz')
    members = read(S / '输入/最新基因机制归属.csv')
    save(members[members.gene_symbol.isin(agreed.index)], '67同向基因既有机制归属.csv')
    for ct in ['Excitatory', 'Inhibitory', 'Astrocyte', 'Microglia_PVM', 'Oligodendrocyte', 'OPC', 'Endothelial', 'Pericyte']:
        z = read(E / '单核结果' / f'SEAAD__{ct}__local_CPS__frontal_prefrontal__regional__genes.csv.gz')
        save(z[z.gene_symbol.isin(agreed.index)].assign(cell_type=ct), f'67基因_{ct}_完整统计.csv')
    spec = {'backend': 'python', 'main_figures': 3, 'supplementary_figures': 2,
        'regions': R, 'cohorts': 20, 'cohort_region_units': 28,
        'source_families': 17, 'profiles': int(roster.profiles.sum()),
        'donor_tissue_records': int(roster.donor_tissue_records.sum()),
        'AD_records': int(roster.AD.sum()), 'control_records': int(roster.control.sum()),
        'common_region_genes': len(common), 'recurrent_at_least_three': len(selected),
        'meta_sumrank_rho': float(rho.statistic), 'shared_three_source_genes': len(core),
        'concordant_three_source_genes': len(agreed), 'up': 56, 'down': 11,
        'interaction_in_current_figures': False, 'Word_changed': False,
        'recalculation_scope': 'Random-effects integration from saved per-cohort DE; source ORA P/BH, recurrence, Top50, intersections and correlations recomputed. Saved SumRank, cameraPR, snRNA models and tSNE coordinates retained.'}
    (O / '核对/复算汇总.json').write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding='utf-8')
    INPUTS.update((B / '论文初稿_0926_Meta八队列_Panel冻结版').glob('*.docx'))
    sources = [dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(INPUTS)]
    pd.DataFrame(sources).to_csv(O / '核对/输入SHA256.csv', index=False, encoding='utf-8-sig')
    print(json.dumps(spec, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
