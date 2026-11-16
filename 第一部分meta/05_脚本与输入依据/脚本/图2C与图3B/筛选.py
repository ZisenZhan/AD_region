"""Freeze a bulk-only display list before any single-nucleus tests."""
from pathlib import Path
import pandas as pd
import numpy as np
O=Path(__file__).resolve().parent
B=O.parent.parent
R=['cerebellum','occipital_visual','frontal_prefrontal','temporal_neocortex','hippocampus','entorhinal_cortex']
AX=['Synapse','Mitochondrial Metabolism','Immune Response','Structural Stabilization','Proteostasis','Lipid Metabolism']
def save(x,file):x.to_csv(O/file,index=False,encoding='utf-8-sig')
def main():
    z=pd.read_csv(O.parent/'源数据/GO_BP方向性结果.csv.gz').query('scope in @R')
    assert not z.duplicated(['scope','go_id']).any()
    sg=B/'第一部分_最新队列六机制重算_20261113'
    ann=pd.read_csv(next(B.rglob('原始19领域_GO_基因对应.csv')),usecols=['ontology','Biodomain','GO_ID']).drop_duplicates()
    ann=ann[ann.ontology.eq('BP')&ann.Biodomain.isin(AX)].groupby('GO_ID').Biodomain.agg(lambda x:tuple(a for a in AX if a in set(x)))
    members=pd.read_csv(sg/'输入/规范GO_BP成员.csv.gz').groupby('go_id').gene_symbol.agg(set)
    q=z.pivot(index='go_id',columns='scope',values='directional_rank_fdr')[R]
    h=z.pivot(index='go_id',columns='scope',values='directional_rank_z')[R]
    a=z.drop_duplicates('go_id').set_index('go_id')[['term']].assign(n_sig=q.lt(.05).sum(axis=1),median_abs_z=h.abs().median(axis=1),complete=q.notna().all(axis=1)&h.notna().all(axis=1),domains=ann)
    a=a.reset_index().sort_values(['n_sig','median_abs_z','go_id'],ascending=[False,False,True])
    used=dict.fromkeys(AX,0);chosen=[]
    reasons=[]
    for t in a.itertuples():
        g=t.go_id
        if not t.complete or not t.n_sig:msg='Incomplete or no significant region'
        elif not isinstance(t.domains,tuple):msg='No direct six-domain BP match'
        elif any(used[v]>=6 for v in t.domains):msg='Domain cap reached'
        else:
            x=members[g]; dup=''
            for j in chosen:
                y=members[j];i=len(x&y)
                if i/len(x|y)>=.5 or i/min(len(x),len(y))>=.85:dup=j;break
            if dup:msg='Redundant with '+dup
            else:
                msg='Selected';chosen.append(g)
                for v in t.domains:used[v]+=1
        reasons.append(msg)
    a['decision']=reasons;a.domains=a.domains.map(lambda x:';'.join(x) if isinstance(x,tuple) else '')
    save(a,'全量筛选审计.csv')
    a=a[a.decision.eq('Selected')].copy()
    a['frontal_q']=a.go_id.map(q.frontal_prefrontal)
    a['frontal_z']=a.go_id.map(h.frontal_prefrontal)
    save(a,'图2C冻结清单.csv')
    f=a[a.frontal_q<.05];save(f,'图3A冻结清单.csv')
    bg=pd.read_csv(sg/'结果/frontal_prefrontal_基因Meta.csv.gz').query('fdr<0.05 and effect!=0').set_index('gene_symbol')
    templates=[(g,y,np.sign(bg.at[y,'effect'])) for g in f.go_id for y in sorted(members[g]&set(bg.index))]
    save(pd.DataFrame(templates,columns=['go_id','gene','weight']),'额叶方向模板.csv')
    print(len(a),len(f),used,flush=True)
if __name__=='__main__':main()
