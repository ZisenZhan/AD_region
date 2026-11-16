"""Crop page whitespace, preserve vector tables, and render review previews."""
from pathlib import Path
import hashlib
import json
import fitz
import pandas as pd

O = Path(__file__).resolve().parents[1]
source = O / '编译/paper_tables.pdf'
final = O / '论文表格_队列与67基因功能.pdf'
doc = fitz.open(source)
assert len(doc) == 2
page_checks = []
for i, page in enumerate(doc):
    rects = [fitz.Rect(b[:4]) for b in page.get_text('blocks') if str(b[4]).strip()]
    rects.extend(d['rect'] for d in page.get_drawings())
    bounds = fitz.Rect(rects[0])
    for rect in rects[1:]:
        bounds |= rect
    assert page.rect.contains(bounds), (i, bounds, page.rect)
    crop = fitz.Rect(bounds.x0-12, bounds.y0-12, bounds.x1+12, bounds.y1+12) & page.rect
    page.set_cropbox(crop)
    page_checks.append(dict(page=i+1, words=len(page.get_text('words')), crop=list(crop)))
doc.set_metadata(dict(title='Brain cohorts and functional annotation of concordant genes',
                     author='', subject='Current frozen bulk and astrocyte results'))
doc.save(final, garbage=4, deflate=True)
doc.close()
with fitz.open(final) as result:
    assert len(result) == 2
    for i, page in enumerate(result):
        page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False).save(O / f'预览/表{i+1}.png')
        assert f'Table {i+1}.' in page.get_text()
    genes = pd.read_csv(O/'源数据/表1_67基因效应及现有机制归属.csv').gene_symbol
    assert all(g in result[0].get_text() for g in genes)
    cohorts = pd.read_csv(O/'源数据/表2_队列脑区样本平台及差异结果.csv')
    assert all(g in result[1].get_text() for g in cohorts.geo.unique())
hashes = pd.read_csv(O/'核对/输入SHA256.csv')
assert all(hashlib.sha256(Path(r.path).read_bytes()).hexdigest() == r.sha256 for r in hashes.itertuples())
(O/'核对/PDF导出验收.json').write_text(json.dumps(dict(pages=page_checks,
    all_67_gene_names_present=True, all_20_GEO_names_present=True,
    source_inputs_unchanged=True, vector_text_preserved=True), indent=2), encoding='utf-8')
print(final)
print(json.dumps(page_checks, indent=2))
