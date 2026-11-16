"""Build the six union sets without selecting terms using bulk or snRNA results."""
from pathlib import Path
import pandas as pd
from runpy import run_path
O=Path(__file__).resolve().parent
B=O.parents[2]
AX=['Synapse','Mitochondrial Metabolism','Immune Response','Structural Stabilization','Proteostasis','Lipid Metabolism']
def save(t,n):t.to_csv(O/n,index=False,encoding='utf-8-sig')
raw=pd.read_csv(next(B.rglob('原始19领域_GO_基因对应.csv')))
a=raw.loc[raw.ontology.eq('BP') & raw.Biodomain.isin(AX)].copy()
names=run_path(str(B/'别名审计与评分轮换_1107/标识与输入.py'))['Names']()
key={s:names.resolve([s],[])[0] for s in a.gene_symbol.dropna().unique()}
a['gene']=a.gene_symbol.map(key)
save(a,'全部BP逐项成员审计.csv.gz')
allgo=a[['Biodomain','GO_ID','GOterm_Name']].drop_duplicates()
save(allgo,'全部GO与机制.csv')
sets=a.loc[a.gene.fillna('').ne(''),['Biodomain','gene']].drop_duplicates()
save(sets,'六机制去重成员.csv')
bg=pd.read_csv(B/'第一部分_最新队列六机制重算_20261113/结果/frontal_prefrontal_基因Meta.csv.gz')
bg=bg.loc[bg.fdr.lt(.05)&bg.effect.ne(0),['gene_symbol','effect']]
t=sets.merge(bg,left_on='gene',right_on='gene_symbol').rename(columns={'Biodomain':'go_id'})
t['weight']=t.effect.gt(0).astype(int)*2-1
save(t[['go_id','gene','weight']],'额叶方向模板.csv')
summary=allgo.groupby('Biodomain').GO_ID.nunique().to_frame('GO').join(sets.groupby('Biodomain').size().rename('genes')).join(t.groupby('go_id').size().rename('template')).reindex(AX).rename_axis('go_id').reset_index()
save(summary,'机制清单.csv')
print(summary.to_string(index=False));print('Unique terms:',allgo.GO_ID.nunique())
