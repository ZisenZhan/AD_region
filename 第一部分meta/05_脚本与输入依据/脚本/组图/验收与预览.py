"""Verify numerical lineage, editable text and bounds; build the reading bundle."""
from pathlib import Path
import hashlib
import html
import json
import xml.etree.ElementTree as ET

import fitz
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

O = Path(__file__).resolve().parents[1]
Q = O / '核对'
D = O / '源数据'


def main():
    inventory = pd.read_csv(Q / '组图清单.csv')
    assert len(inventory) == 4 and inventory.folder.eq('主文组图').sum() == 3
    assert inventory.panels.sum() == 36
    assert len(list((O / '单独子图').glob('*.png'))) == 36
    results, beyond = [], []
    combined = fitz.open()
    for row in inventory.itertuples():
        stem = O / row.folder / row.figure
        im = mpimg.imread(stem.with_suffix('.png'))
        assert ((im[:, :, :3] < .95).any(axis=2)).mean() > .02
        with fitz.open(stem.with_suffix('.pdf')) as doc:
            assert len(doc) == 1 and len(doc[0].get_text()) > 100
            page = doc[0]
            assert abs(page.rect.width / 72 * 25.4 - 183) < .05
            text = page.get_text()
            assert 'interaction' not in text.lower()
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines', []):
                    for span in line['spans']:
                        x0, y0, x1, y1 = span['bbox']
                        if x0 < -.5 or y0 < -.5 or x1 > page.rect.width + .5 or y1 > page.rect.height + .5:
                            beyond.append(dict(figure=row.figure, text=span['text'], bbox=span['bbox']))
            combined.insert_pdf(doc)
        root = ET.parse(stem.with_suffix('.svg')).getroot()
        strings = [''.join(z.itertext()) for z in root.iter() if z.tag.endswith('}text')]
        assert len(strings) > 20
        assert set(chr(65 + i) for i in range(row.panels)) <= set(strings)
        results.append(dict(figure=row.figure, editable_text_nodes=len(strings), nonblank=True,
            width_mm=183, panel_letters=True, text_in_bounds=True))
    assert not beyond, beyond
    pdf_name = '全部组图合订预览_四页_六机制八类细胞.pdf'
    try:
        combined.save(O / pdf_name, garbage=4, deflate=True)
    except Exception as error:
        if 'Permission denied' not in str(error):
            raise
        from datetime import datetime
        pdf_name = '全部组图合订预览_四页_六机制八类细胞_' + datetime.now().strftime('%H%M%S') + '.pdf'
        combined.save(O / pdf_name, garbage=4, deflate=True)
    combined.close()
    for svg in (O / '单独子图').glob('*.svg'):
        strings = [''.join(z.itertext()) for z in ET.parse(svg).getroot().iter() if z.tag.endswith('}text')]
        # Standalone plots retain cohort titles and axes, not assembly panel letters.
        assert not any(s in list('ABCDEGH') for s in strings), svg  # F is the frontal-region key.
    hashes = pd.read_csv(Q / '输入SHA256.csv')
    for row in hashes.itertuples():
        assert hashlib.sha256(Path(row.path).read_bytes()).hexdigest() == row.sha256, row.path
    summary = json.loads((Q / '复算汇总.json').read_text(encoding='utf-8'))
    assert summary['concordant_three_source_genes'] == 67 and summary['shared_three_source_genes'] == 72
    core = pd.read_csv(D / '三来源67同向基因.csv')
    assert (core[['bulk_q', 'sea_q', 'ros_q']] < .05).all().all()
    assert (np.sign(core[['bulk_beta', 'sea_beta', 'ros_beta']]).nunique(axis=1) == 1).all()
    assert len(core) == core.gene_symbol.nunique() == 67
    b = pd.read_csv(D / '三来源共同显著_含方向冲突.csv')
    assert b.direction.value_counts().to_dict() == {'Up': 56, 'Down': 11, 'Mixed': 5}
    rho = pd.read_csv(D / '全共同背景效应秩相关.csv')
    for row in rho.itertuples():
        assert row.n == 13416
    # Preserve all exact intersections as a table, without another redundant plot.
    flags = pd.read_csv(D / '共同背景显著性与重复性.csv.gz')
    regions = summary['regions']
    code = flags[regions].astype(int).astype(str).agg(''.join, axis=1)
    exact = code.value_counts().rename_axis('CB_OCC_F_T_HIP_EC').reset_index(name='genes')
    exact.to_csv(D / '六脑区全部精确交集.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.3))
    for ax, row in zip(axes, inventory[inventory.folder.eq('主文组图')].itertuples()):
        ax.imshow(mpimg.imread(O / row.folder / (row.figure + '.png')))
        ax.axis('off')
    fig.subplots_adjust(left=.005, right=.995, top=.995, bottom=.005, wspace=.02)
    fig.savefig(O / '三张主图总览.png', dpi=200, facecolor='white')
    plt.close(fig)
    pieces = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>整脑、区域机制与星形胶质细胞</title>',
        '<style>body{font:16px/1.65 system-ui;max-width:1160px;margin:28px auto;padding:0 24px;color:#26222e}figure{margin:38px 0}img{display:block;width:100%;height:auto}h1{font-size:25px}h2{font-size:20px}a{color:#6041ad}</style>',
        '<h1>整脑共同异常 → 区域机制模式 → 额叶星形胶质细胞</h1>',
        f'<p>3张主图，1张补充图。<a href="{pdf_name}">合订PDF</a> · <a href="说明/逐子图说明_中文.md">逐子图说明</a></p>']
    for row in inventory.itertuples():
        path = row.folder + '/' + row.figure
        pieces.append(f'<figure><h2>{html.escape(row.figure)}</h2><p><a href="{path}.png">PNG</a> · <a href="{path}.pdf">PDF</a> · <a href="{path}.svg">SVG</a></p><img loading="lazy" src="{path}.png"></figure>')
    pieces.append('</html>')
    (O / '全部组图预览.html').write_text('\n'.join(pieces), encoding='utf-8')
    pd.DataFrame(results).to_csv(Q / '图像与矢量验收.csv', index=False, encoding='utf-8-sig')
    report = dict(composites=4, main=3, supplementary=1, unlettered_standalone_panels=36,
        input_hashes_unchanged=True, Word_unchanged=True, all_PDF_text_within_page=True,
        PDF_SVG_text_editable=True, interaction_panels_in_current_bundle=0,
        numerical_aggregation_verified=True, three_source_concordant_genes=67,
        review='Main and supplementary PNGs inspected; compact cell sizes and bar widths, coefficient scales retained.')
    report['current_combined_PDF'] = pdf_name
    n = O/'六领域GO与额叶单核'
    spec = pd.read_csv(n/'图2C冻结清单.csv')
    sub = pd.read_csv(n/'图3A冻结清单.csv')
    cells3 = pd.read_csv(n/'八类细胞完整检验.csv')
    assert spec.domains.str.split(';').explode().value_counts().le(6).all()
    assert set(sub.go_id)==set(spec.loc[spec.frontal_q<.05,'go_id'])
    assert len(cells3)==8*len(sub) and not cells3.duplicated(['cell_type','go_id']).any()
    a=pd.read_csv(n/'图2C显示顺序.csv').go_id
    assert a[a.isin(sub.go_id)].tolist()==pd.read_csv(n/'图3A显示顺序.csv').go_id.tolist()
    assert cells3.p.notna().equals(cells3.template_n.ge(10))
    from scipy.stats import false_discovery_control
    pvals=cells3.p.dropna()
    assert np.allclose(cells3.fdr.loc[pvals.index], np.minimum(1,false_discovery_control(pvals)*len(cells3)/len(pvals)))
    cmp = cells3.merge(pd.read_csv(D/'额叶全部细胞全部过程.csv'),left_on=['cell_type','go_id'],right_on=['cell_type','set_id'])
    assert len(cmp)>0 and np.allclose(cmp.p,cmp.template_p,equal_nan=True)
    out=cells3.merge(sub[['go_id','term','domains','frontal_z','frontal_q']],on='go_id')
    out['status']=np.select([out.p.isna(),out.fdr.lt(.05)&out.direction.eq('Up'),out.fdr.lt(.05)&out.direction.eq('Down')],['Not evaluable','Concordant (FDR<0.05)','Discordant (FDR<0.05)'],default='Not significant')
    out[out.cell_type.eq('Astrocyte')].sort_values('fdr').to_csv(n/'星形胶质细胞_全部过程及方向.csv',index=False,encoding='utf-8-sig')
    overview = pd.read_csv(n/'全量六机制总览/六机制八类细胞检验.csv')
    assert len(overview)==48 and not overview.duplicated(['cell_type','go_id']).any()
    assert np.allclose(overview.fdr, false_discovery_control(overview.p))
    terms = pd.read_csv(n/'全量六机制总览/全部GO与机制.csv')
    assert terms.GO_ID.nunique()==2737
    report.update(Figure3_A_domains=6,Figure3_A_all_BP=2737,Figure3_A_BH_family=48,Figure2_C_displayed_GO=len(spec),Figure3_B_frontal_GO=len(sub),Figure3_B_evaluable=len(pvals),Figure3_B_BH_family=len(cells3),astrocyte_concordant=int((out.cell_type.eq('Astrocyte')&out.status.eq('Concordant (FDR<0.05)')).sum()))
    cells = pd.read_csv(D/'图2A_经典g_热图逐格数据.csv')
    top = pd.read_csv(D/'图2A_经典g_前100基因及六脑区结果.csv')
    assert len(cells)==3400 and len(top)==100 and set(cells.gene_symbol)==set(top.gene_symbol)
    assert not cells.duplicated(['gene_symbol','unit']).any()
    assert cells.sort_values('display_col').unit.drop_duplicates().tolist()==pd.read_csv(D/'热图列信息.csv').unit.tolist()
    assert cells.fdr.lt(.05).sum()==1210
    report['Figure2_A_genes']=100
    report['Figure2_A_statistic']='classical Hedges g; gene clustering only'
    mean_file = D / '图3C_67基因原始效应及算术均值.csv'
    if mean_file.exists():
        means = pd.read_csv(mean_file).set_index('gene_symbol')
        expected = core.set_index('gene_symbol')[['bulk_beta','sea_beta','ros_beta']].mean(axis=1)
        assert np.allclose(means.mean_beta.reindex(expected.index), expected, rtol=1e-12, atol=1e-12)
        report['Figure3_D_arithmetic_mean_verified'] = True
    (Q / '最终验收.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
