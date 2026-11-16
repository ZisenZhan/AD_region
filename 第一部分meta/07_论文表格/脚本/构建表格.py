"""Build publication tables from frozen current analysis and sample provenance."""
from pathlib import Path
import csv
import gzip
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

O = Path(__file__).resolve().parents[1]
B = O.parent
ROOT = B.parents[1]
F = B / '整脑_区域机制_额叶星形胶质_紧凑组图_20261116/源数据'
DB = B / '别名审计与评分轮换_1107/训练与评测缓存'
EXTRA = B / '候选队列核查扩增_1107/122063_37263'
CURRENT = B / '第一部分_最新队列六机制重算_20261113'
EVIDENCE = B / 'TREATAD六领域_并列保留_Panel六基因对照_1105/源数据/分机制成员_GO证据.csv'
INPUTS = set()
REGIONS = ['frontal_prefrontal','temporal_neocortex','hippocampus','entorhinal_cortex','occipital_visual','cerebellum']
NAMES = dict(zip(REGIONS, ['Frontal/prefrontal','Temporal neocortex','Hippocampus','Entorhinal','Occipital/visual','Cerebellum']))
DOMAINS = ['Synapse','Mitochondrial Metabolism','Immune Response','Structural Stabilization','Proteostasis','Lipid Metabolism']


def read(p, **kw):
    INPUTS.add(p)
    return pd.read_csv(p, float_precision='round_trip', **kw)


def save(d, name):
    d.to_csv(O / '源数据' / name, index=False, encoding='utf-8-sig', float_format='%.17g')


def tex(s):
    replace = {'&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    return ''.join(replace.get(c,c) for c in str(s))


def header_metadata(path):
    INPUTS.add(path)
    opener = gzip.open if path.suffix == '.gz' else open
    fields = {}
    with opener(path, 'rt', encoding='utf-8', errors='replace') as f:
        for row in csv.reader(f, delimiter='\t'):
            if not row:
                continue
            if row[0] == '!series_matrix_table_begin':
                break
            if row[0] in ['!Series_platform_id','!Series_type','!Sample_geo_accession','!Sample_platform_id']:
                fields.setdefault(row[0], []).extend(row[1:])
    return fields


def platform_for_geo(geo):
    folder = ROOT / 'GSE_datasets' / geo
    if geo == 'GSE122063':
        INPUTS.add(EXTRA / '队列裁定.md')
        return ['GPL16699'], 'Microarray', str(EXTRA / '队列裁定.md')
    if geo == 'GSE37263':
        path = EXTRA / '原始下载/GSE37263_series_matrix.txt.gz'
        files = [path]
    elif geo == 'GSE84422':
        files = [folder / (geo + '-' + platform + '_series_matrix.txt') for platform in ['GPL96','GPL97']]
    else:
        files = sorted(folder.glob('*series_matrix*.txt')) or sorted(folder.glob('*series_matrix*.gz'))
    assert files, geo
    plats, types = set(), set()
    for file in files:
        h = header_metadata(file)
        # Multi-platform series headers list unused platforms too.
        platform_field = '!Sample_platform_id' if geo == 'GSE84422' else '!Series_platform_id'
        plats.update(h.get(platform_field, []))
        types.update(h.get('!Series_type', []))
    assert plats, geo
    if any('high throughput sequencing' in t for t in types):
        assay = 'RNA-seq'
    else:
        assert types == {'Expression profiling by array'}, (geo, types)
        assay = 'Microarray'
    return sorted(plats), assay, ';'.join(str(p) for p in files)


def main():
    for folder in ['源数据','tex','编译','预览','核对']:
        (O / folder).mkdir(parents=True, exist_ok=True)
    genes = read(F / '三来源67同向基因.csv')
    mapping = read(F / '67同向基因既有机制归属.csv').set_index('gene_symbol').domains
    assert len(genes) == genes.gene_symbol.nunique() == 67
    assert genes.direction.value_counts().to_dict() == {'Up':56,'Down':11}
    detailed = genes.copy()
    detailed['domains'] = detailed.gene_symbol.map(mapping).fillna('Not assigned')
    save(detailed, '表1_67基因效应及现有机制归属.csv')
    rows = []
    for domain in DOMAINS + ['Not assigned']:
        sub = detailed[detailed.domains.str.split(';').map(lambda x: domain in x)]
        rows.append(dict(domain=domain,n=len(sub),up=int(sub.direction.eq('Up').sum()),
            down=int(sub.direction.eq('Down').sum()),genes=';'.join(sorted(sub.gene_symbol))))
    summary = pd.DataFrame(rows)
    assert len(mapping) == 51 and summary.iloc[-1]['n'] == 16
    assert set(';'.join(summary.genes).split(';')) == set(genes.gene_symbol)
    save(summary, '表1_功能归类与基因名单.csv')
    evidence = read(EVIDENCE)
    evidence = evidence[evidence.gene_symbol.isin(genes.gene_symbol)].copy()
    evidence['current_domains'] = evidence.gene_symbol.map(mapping)
    evidence['supports_current_domain'] = evidence.apply(
        lambda row: row.Biodomain in str(row.current_domains).split(';'),axis=1)
    evidence = evidence[evidence.supports_current_domain]
    save(evidence, '表1_可追溯GO条目证据.csv')
    coverage = evidence.groupby('gene_symbol').Biodomain.agg(lambda x: set(x))
    for gene, domains in mapping.items():
        assert gene in coverage and set(domains.split(';')) <= coverage[gene], (gene, domains)
    roster = read(F / '最新队列与样本覆盖.csv')
    long = read(CURRENT / '输入/规范基因队列差异与模型衍生g.csv.gz')
    assert len(roster) == 28 and roster.geo.nunique() == 20
    assert not long.duplicated(['unit','gene_symbol']).any()
    family_keys = sorted(roster.family.unique())
    family_ids = {key:f'F{i+1:02d}' for i,key in enumerate(family_keys)}
    family_rows = []
    for key in family_keys:
        sub = roster[roster.family.eq(key)]
        family_rows.append(dict(family_id=family_ids[key],source_family_key=key,
            included_GEO=';'.join(sorted(sub.geo.unique())),cohort_region_contributions=len(sub)))
    save(pd.DataFrame(family_rows),'表2_来源家族对应.csv')
    platforms = {geo: platform_for_geo(geo) for geo in sorted(roster.geo.unique())}
    save(pd.DataFrame([dict(geo=g,platforms=';'.join(v[0]),assay=v[1],source=v[2]) for g,v in platforms.items()]),
        '表2_平台信息来源.csv')
    records, samples = [], []
    for region in REGIONS:
        for row in roster[roster.region.eq(region)].sort_values('geo').itertuples():
            path = DB / (row.unit + '_md.pkl')
            INPUTS.add(path)
            md = pd.read_pickle(path)
            assert md.groupby('key').y.nunique().max() == 1
            source_samples = md.source_sample_ids.astype(str).str.split(';').explode()
            assert source_samples.str.startswith('GSM').all(), row.unit
            n_source = source_samples.nunique()
            n_analysis = md.key.nunique()
            ad = int(md.groupby('key').y.first().sum())
            control = n_analysis-ad
            assert (n_analysis,ad,control) == (row.donor_tissue_records,row.AD,row.control)
            de = long[long.unit.eq(row.unit)].copy()
            de['BH_FDR'] = false_discovery_control(de.p_value.to_numpy())
            hit = de.BH_FDR.lt(.05)
            platforms_geo, assay, _ = platforms[row.geo]
            handling = 'No within-unit merge in retained records'
            if n_source > n_analysis:
                handling = 'Repeated donor records aggregated within region'
            if row.geo == 'GSE122063':
                handling = 'Technical replicates/dye swaps averaged within donor and region'
            elif row.geo == 'GSE84422':
                handling = 'Matched GPL96/GPL97 measurements combined; each donor counted once'
            elif row.geo == 'GSE29378':
                handling = 'CA1/CA3 subfield records aggregated by donor within the hippocampal group'
            records.append(dict(region=region,brain_region=NAMES[region],geo=row.geo,assay=assay,
                platform=' + '.join(platforms_geo),AD=ad,control=control,source_GSM_records=n_source,
                cached_input_rows=len(md),analysed_donor_region_records=n_analysis,
                merged_records=n_source-n_analysis,merge_handling=handling,genes_tested=len(de),
                nominal_DEGs=int(de.p_value.lt(.05).sum()),BH_DEGs=int(hit.sum()),
                BH_up=int((hit&de.logFC.gt(0)).sum()),BH_down=int((hit&de.logFC.lt(0)).sum()),
                family_id=family_ids[row.family],source_family_key=row.family))
            for item in md.itertuples():
                for gsm in str(item.source_sample_ids).split(';'):
                    samples.append(dict(unit=row.unit,geo=row.geo,region=region,GSM=gsm,
                        donor_key=str(item.key),group='AD' if item.y==1 else 'Control',
                        family_id=family_ids[row.family]))
    table = pd.DataFrame(records)
    table.insert(0,'No',np.arange(1,len(table)+1))
    assert int(table.AD.sum()) == 1151 and int(table.control.sum()) == 890
    assert int(table.analysed_donor_region_records.sum()) == 2041
    assert int(table.cached_input_rows.sum()) == 2116
    assert table.family_id.nunique() == 17
    assert (table.BH_up+table.BH_down).equals(table.BH_DEGs)
    assert int(table.genes_tested.sum()) == len(long)
    save(table, '表2_队列脑区样本平台及差异结果.csv')
    sample_table = pd.DataFrame(samples).drop_duplicates()
    assert len(sample_table) == int(table.source_GSM_records.sum())
    assert not sample_table.duplicated(['unit','GSM']).any()
    assert sample_table.groupby(['unit','donor_key'])['group'].nunique().le(1).all()
    save(sample_table, '表2_纳入GSM与供体聚合对应.csv')
    functional_tex(summary)
    cohort_tex(table)
    audit = dict(cohort_region_contributions=28, GEO=20, conservative_source_families=17,
        analysed_donor_region_records=2041, AD=1151, control=890,
        source_GSM_records=int(table.source_GSM_records.sum()),
        merged_source_records=int(table.merged_records.sum()),cached_input_rows=2116,
        genes_in_functional_table=67,mapped_genes=51,not_assigned=16,
        membership_counts=summary.set_index('domain')['n'].to_dict(),
        no_outlier_counts_inferred=True,no_probe_counts_or_sequencing_depth_inferred=True,
        original_models_or_figures_modified=False)
    (O/'核对/表格数据验收.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    save_hashes = [dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                   for p in sorted(INPUTS)]
    pd.DataFrame(save_hashes).to_csv(O/'核对/输入SHA256.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


def functional_tex(summary):
    lines=[r'\fontsize{10}{12}\selectfont',
        r'\textbf{Table 1. Functional annotation of 67 concordant genes across frontal bulk and two astrocyte sources.}\par\vspace{5pt}',
        r'\fontsize{8.6}{10.7}\selectfont',
        r'\begin{tabularx}{\linewidth}{|L{34mm}|C{20mm}|Y|}\hline',
        r'\textbf{Biological domain} & \textbf{n (up/down)} & \textbf{Gene list} \\ \hline']
    for row in summary.itertuples():
        name='Not assigned to the six domains' if row.domain=='Not assigned' else row.domain
        genes=', '.join(row.genes.split(';'))
        lines.append(tex(name)+' & '+f'{row.n} ({row.up}/{row.down})'+' & '+tex(genes)+r' \\ \hline')
    lines += [r'\end{tabularx}\par\vspace{5pt}',r'\fontsize{8}{10}\selectfont',
        r'All genes have FDR $<0.05$ in frontal bulk Meta, SEA-AD frontal astrocyte local-CPS associations, and ROSMAP-427 prefrontal astrocyte gpath associations, with concordant signs (56 positive; 11 negative). ',
        r'Current six-domain assignments use BP/MF support, CC fallback when BP/MF support is unavailable, and retain tied highest-scoring domains. Fifty-one genes map to at least one domain; 16 are unassigned in this annotation. Membership counts overlap. ',
        r'BP, biological process; MF, molecular function; CC, cellular component.']
    (O/'tex/functional_table.tex').write_text('\n'.join(lines),encoding='utf-8')


def cohort_tex(table):
    lines=[r'\fontsize{10}{12}\selectfont',
        r'\textbf{Table 2. Included brain transcriptomic cohorts, donor--region records, platforms and within-cohort differential expression.}\par\vspace{5pt}',
        r'\fontsize{7.7}{9.8}\selectfont',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.10}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}r L{28mm} L{23mm} L{17mm} L{25mm} r r r r r r r r l@{}}',
        r'\toprule',
        r'\textbf{No.} & \textbf{Brain region} & \textbf{GEO} & \textbf{Assay} & \textbf{Platform} & \textbf{AD} & \textbf{Ctrl} & \textbf{\shortstack{Merged\\records}} & \textbf{\shortstack{Genes\\tested}} & \textbf{\shortstack{DEGs\\\textit{P}<0.05}} & \textbf{\shortstack{DEGs\\FDR<0.05}} & \textbf{Up} & \textbf{Down} & \textbf{Family} \\',
        r'\midrule']
    previous=None
    for row in table.itertuples():
        if previous is not None and previous!=row.region:
            lines.append(r'\addlinespace[3pt]')
        previous=row.region
        vals=[row.No,row.brain_region,row.geo,row.assay,row.platform,row.AD,row.control,row.merged_records,
            f'{row.genes_tested:,}',f'{row.nominal_DEGs:,}',f'{row.BH_DEGs:,}',f'{row.BH_up:,}',f'{row.BH_down:,}',row.family_id]
        lines.append(' & '.join(tex(v) for v in vals)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular*}\par\vspace{5pt}',r'\fontsize{7.6}{9.6}\selectfont',
        r'AD/control counts are analysed donor--region records after within-unit aggregation. The 28 contributions comprise 20 GEO series, 17 conservative source families, and 2,041 donor--region records (1,151 AD; 890 control). ',
        r'Merged records equal retained source-GSM measurements minus analysed donor--region records; they are not outlier exclusions. GSE122063 combines technical/dye-swap replicates, GSE84422 combines paired GPL96/GPL97 measurements, and GSE29378 combines CA1/CA3 records within donors. ',
        r'Other repeated donor records are aggregated within region. Genes tested are canonical symbols with valid model estimates. Nominal DEGs use two-sided \textit{P}<0.05; FDR DEGs use Benjamini--Hochberg FDR<0.05 across all tested genes in each cohort--region unit. Up/down are the FDR-significant AD--control directions. ',
        r'Family identifiers follow the current conservative grouping; the corresponding source list is provided with the table data. Donors sampled in different regions can contribute more than one donor--region record.']
    (O/'tex/cohort_table.tex').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
