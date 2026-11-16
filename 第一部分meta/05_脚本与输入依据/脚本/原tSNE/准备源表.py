"""Prepare outcome-blind embeddings and direct plotting tables; no model refitting."""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

O=Path(__file__).resolve().parents[1];B=O.parent;S=B/'第一部分_最新队列六机制重算_20261113'
S=Path(os.environ.get('AD_META_SOURCE',S))
sys.path.insert(0,str(B/'别名审计与评分轮换_1107'))
import 检验与轮换 as old
for n in ['源数据','主文组图','补充组图','单独子图','核对','说明','参考']: (O/n).mkdir(exist_ok=True)
SRC=O/'源数据';CHECK=O/'核对';SEED=20261113
def save(z,name):pd.DataFrame(z).to_csv(SRC/name,index=False,encoding='utf-8-sig',float_format='%.17g')
def embedding(geo,units):
    xs=[];ms=[]
    for uid in units:
        region=uid.split('__')[0];m=pd.read_pickle(old.DB/(uid+'_md.pkl'));parts=[]
        for f in sorted(old.CACHE.glob(uid+'__*_rank.pkl')):
            x=pd.read_pickle(f)
            labels=m.key if geo in ['GSE122063','GSE37263','GSE159699'] else m.source_sample_ids.map(lambda s:next(k for k in str(s).split(';') if k in x.index))
            x=x.loc[labels].copy();x.index=m.index;parts.append(x)
        assert parts,uid
        x=parts[0] if len(parts)==1 else pd.concat(parts,axis=1).T.groupby(level=0,sort=False).median().T
        x=x.groupby(m.key).mean();meta=m.groupby('key').first().loc[x.index]
        meta['key']=meta.index.astype(str)
        x.index=region+'|'+x.index.astype(str);meta.index=x.index
        xs.append(x);ms.append(meta.assign(region=region))
    common=sorted(set.intersection(*(set(x.columns) for x in xs)))
    x=pd.concat([v[common] for v in xs]);m=pd.concat(ms).loc[x.index]
    var=x.var().sort_values(ascending=False);genes=var[var>0].head(2000).index
    x=x[genes].fillna(x[genes].median())
    assert x.index.is_unique and x.notna().all().all()
    with threadpool_limits(limits=1):
        pc=PCA(n_components=min(30,len(x)-1,len(genes)),random_state=SEED).fit_transform(x)
        per=min(30.,max(2.,(len(x)-1)/3))
        xy=TSNE(n_components=2,perplexity=per,max_iter=1000,init='pca',learning_rate='auto',random_state=SEED,n_jobs=1).fit_transform(pc)
    z=m[['key','y','region']].copy().assign(geo=geo,tsne1=xy[:,0],tsne2=xy[:,1])
    meta=dict(geo=geo,n=len(x),AD=int(m.y.sum()),control=int((1-m.y).sum()),regions=m.region.nunique(),
              common_genes=len(common),selected_genes=len(genes),perplexity=per,seed=SEED,
              PC_components=pc.shape[1],PC_definition='PCA of per-sample within-profile gene percentiles; no per-region column recalibration')
    return z,meta,list(genes)

def main():
    roster=pd.read_csv(S/'输入/最新全发现队列清单.csv');save(roster,'队列与脑区.csv')
    tasks=[(g,sorted(z.unit)) for g,z in roster.groupby('geo')]
    result=Parallel(n_jobs=4,backend='threading')(delayed(embedding)(g,u) for g,u in tasks)
    save(pd.concat([z[0] for z in result]),'全部GEO样本级tSNE.csv.gz')
    save([z[1] for z in result],'tSNE参数与人数.csv')
    (SRC/'tSNE变异基因.json').write_text(json.dumps({a[1]['geo']:a[2] for a in result},indent=2),encoding='utf-8')
    keys=['whole_brain','cerebellum','occipital_visual','frontal_prefrontal','temporal_neocortex','hippocampus','entorhinal_cortex']
    summary=[]
    for scope in keys:
        m=pd.read_csv(S/'结果'/(scope+'_基因Meta.csv.gz'),float_precision='round_trip')
        sr=pd.read_csv(S/'结果'/(scope+'_完整SumRank.csv.gz'),float_precision='round_trip')
        save(m,scope+'_Meta.csv.gz');save(sr,scope+'_SumRank.csv.gz')
        summary.append(dict(scope=scope,genes=len(m),up=int(((m.fdr<.05)&(m.effect>0)).sum()),down=int(((m.fdr<.05)&(m.effect<0)).sum())))
    save(summary,'各范围显著基因数量.csv')
    for p in [S/'结果/六脑区共同背景显著性.csv.gz',S/'输入/规范基因队列差异与模型衍生g.csv.gz',S/'结果/六机制cameraPR区域检验.csv',
              S/'结果/GO_BP方向性结果.csv.gz',S/'结果/GO_BP_共同基因ORA.csv',S/'结果/KEGG_共同基因ORA.csv',
              S/'结果/分脑区效应量与I2变化.csv',S/'结果/标准化g尺度I2敏感性汇总.csv',S/'结果/逐基因分层前后配对.csv.gz',
              S/'结果/新旧基因Meta结论对照.csv',S/'交互结果/重复脑区对汇总.csv',S/'交互结果/六机制cameraPR交互整合.csv',
              S/'交互结果/原六个突触GO_不受广泛扫描大小阈值限制.csv']:
        save(pd.read_csv(p,float_precision='round_trip'),p.name)
    sources=list(S.rglob('*.csv*'))
    sources += [old.DB/(uid+'_md.pkl') for uid in roster.unit]
    sources += [p for uid in roster.unit for p in old.CACHE.glob(uid+'__*_rank.pkl')]
    save_manifest=pd.DataFrame([dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(set(sources))])
    save_manifest.to_csv(CHECK/'上游源文件SHA256.csv',index=False,encoding='utf-8-sig')
    print('Prepared',len(tasks),'GEO embeddings and frozen plotting tables.',flush=True)

if __name__=='__main__':main()
