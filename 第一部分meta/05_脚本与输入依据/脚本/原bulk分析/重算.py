"""Rebuild discovery meta-analysis from canonical cohort DE and current domains."""
from pathlib import Path
import os
for v in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[v]='1'
import sys, json, hashlib, importlib.util, itertools
import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from threadpoolctl import threadpool_limits

OUT=Path(__file__).resolve().parents[1]
BASE=OUT.parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(BASE/'别名审计与评分轮换_1107'))
import 检验与轮换 as old
DB=old.DB
for name in ['输入','结果','图件','核对','交互输入','交互结果']: (OUT/name).mkdir(exist_ok=True)
FILES=set()
ROSTER={
    'frontal_prefrontal':'118553 122063 150696 185909 261817 313307 33000 36980 48350 53697',
    'temporal_neocortex':'118553 122063 132903 36980 37263',
    'hippocampus':'1297 173954 29378 318560 36980 48350',
    'entorhinal_cortex':'118553 48350','occipital_visual':'203206 44771 84422','cerebellum':'118553 44768'}
def frozen_units(available):
    expected={r+'__GSE'+g for r,gs in ROSTER.items() for g in gs.split()}
    assert len(expected)==28 and expected<=set(available),expected-set(available)
    return sorted(expected)
def save(x,name): pd.DataFrame(x).to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')
def read(p):
    FILES.add(Path(p)); return pd.read_pickle(p)
def module(n):
    p=next((ROOT/'region_ad_analysis').glob(n+'_*.py'))
    spec=importlib.util.spec_from_file_location('refresh_'+n,p); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
meta=module('05'); pw=module('07')
def family(g): return '+'.join(sorted(old.base.blocked(g)))

def integrate(long):
    # Collapse correlated tissues/series conservatively: no precision gain from averaging.
    d=long.assign(se2=long.se**2).groupby(['family','gene_symbol'],sort=True).agg(
        logFC=('logFC','mean'),se2=('se2','mean'),g=('g','mean')).reset_index()
    d['se']=np.sqrt(d.se2);d['p_value']=2*stats.norm.sf(abs(d.logFC/d.se))
    e=d.pivot(index='gene_symbol',columns='family',values='logFC')
    s=d.pivot(index='gene_symbol',columns='family',values='se').reindex_like(e)
    g=d.pivot(index='gene_symbol',columns='family',values='g').reindex_like(e)
    y=e.to_numpy(); se=s.to_numpy(); ok=np.isfinite(y)&np.isfinite(se)&(se>0)
    k=ok.sum(axis=1); w=np.where(ok,1/se**2,0); yy=np.where(ok,y,0)
    sw=w.sum(axis=1); fixed=(w*yy).sum(axis=1)/sw
    q=(w*np.where(ok,(y-fixed[:,None])**2,0)).sum(axis=1)
    c=sw-(w*w).sum(axis=1)/sw
    tau=np.maximum(0,np.divide(q-(k-1),c,out=np.zeros_like(q),where=c>0))
    wr=np.where(ok,1/(se**2+tau[:,None]),0); sr=wr.sum(axis=1)
    est=(wr*yy).sum(axis=1)/sr; ese=np.sqrt(1/sr); p=2*stats.norm.sf(abs(est/ese))
    i2=np.maximum(0,np.divide(q-(k-1),q,out=np.zeros_like(q),where=q>0))*100
    i2[k<=1]=0
    ans=pd.DataFrame(dict(gene_symbol=e.index,k=k,effect=est,se=ese,p=p,fdr=meta.bh_adjust(pd.Series(p)),
        ci_low=est-1.96*ese,ci_high=est+1.96*ese,I2=i2,tau2=tau,median_g_star=g.median(axis=1).to_numpy(),
        direction_agreement=np.maximum((y>0).sum(axis=1),(y<0).sum(axis=1))/k))
    return ans,d

def raw_unit(uid, manifest):
    region,geo=uid.split('__'); md=read(DB/(uid+'_md.pkl'))
    if geo=='GSE122063':
        p=old.CACHE/(uid+'_expression.tsv.gz'); FILES.add(p)
        x=pd.read_csv(p,sep='\t',index_col=0).T
        x.index=x.index.astype(str); m=md.groupby('key').first().loc[x.index]
    else:
        row=manifest.loc[manifest.unit_id.eq(uid)].iloc[0]
        p=Path(row.matrix_path); FILES.add(p)
        raw=pd.read_csv(p,dtype={pd.read_csv(p,nrows=0).columns[0]:str}).set_index(pd.read_csv(p,nrows=0).columns[0])
        raw.index=raw.index.map(old.base.idmap.clean)
        mapping=old.base.idmap.AUD/(geo+'__'+row.component+'_全部特征.csv.gz')
        FILES.add(mapping); mp=pd.read_csv(mapping,keep_default_na=False,dtype=str).set_index('feature_id').gene_symbol
        gs=raw.index.map(mp).fillna(''); x=raw.loc[gs!=''].groupby(gs[gs!='']).median().T
        labels=md.source_sample_ids.map(lambda s: next(t for t in str(s).split(';') if t in x.index))
        x=x.loc[labels].copy();x.index=md.index
        x=x.groupby(md.key).mean();m=md.groupby('key').first().loc[x.index]
    assert x.index.equals(m.index) and m.y.nunique()==2
    outmeta=pd.DataFrame(dict(sample_id=x.index,donor_id=x.index,standard_group=np.where(m.y==1,'AD','control'),
                              region_slug=region,age_standard=np.nan,sex_standard='Missing'))
    sm=ROOT/'outputs/ad_region_analysis/sample_manifests'/(uid+'_samples.csv')
    if sm.exists():
        FILES.add(sm); ss=pd.read_csv(sm,dtype=str).set_index('sample_id')
        for i,donor in enumerate(x.index):
            source=str(m.loc[donor,'source_sample_ids']).split(';')[0]
            if source in ss.index:
                for c in ['age_standard','sex_standard']:
                    if c in ss: outmeta.loc[i,c]=pd.to_numeric(ss.loc[source,c],errors='coerce') if c=='age_standard' else ss.loc[source,c]
    if geo=='GSE122063':
        p=old.base.idmap.NEW/'输入'/(uid+'__meta.csv');FILES.add(p)
        ss=pd.read_csv(p,dtype=str).set_index('key')
        outmeta['age_standard']=outmeta.donor_id.map(ss.age)
        outmeta['sex_standard']=outmeta.donor_id.map(ss.sex)
    x.T.to_csv(OUT/'交互输入'/(uid+'_expression.csv.gz'),encoding='utf-8')
    outmeta.to_csv(OUT/'交互输入'/(uid+'_meta.csv'),index=False,encoding='utf-8-sig')
    return x,m

def main():
    av=read(DB/'可测.pkl'); units=frozen_units(av.columns)
    domains_path=old.CACHE/'机制归属.csv.gz';FILES.add(domains_path)
    domains=pd.read_csv(domains_path).set_index('gene_symbol').domains
    save(domains.reset_index(),'输入/最新基因机制归属.csv')
    memberships={a:set(domains.index[domains.map(lambda z:a in z.split(';'))]) for a in old.AXES}
    save([dict(mechanism=a,gene_symbol=g) for a,gs in memberships.items() for g in sorted(gs)],'输入/六机制基因成员.csv.gz')
    mf_path=BASE/'audit/fixed_region_input_manifest.csv';FILES.add(mf_path)
    manifest=pd.read_csv(mf_path)
    frames=[]; roster=[]
    for uid in units:
        r,g=uid.split('__'); md=read(DB/(uid+'_md.pkl')); tm=read(DB/(uid+'_info.pkl'))
        z=read(DB/(uid+'_de.pkl')).copy()
        if g=='GSE185909':
            x,m=raw_unit(uid,manifest);a=x.loc[m.y.eq(1)]; b=x.loc[m.y.eq(0)]
            z=pd.DataFrame(dict(logFC=a.mean()-b.mean(),se=np.sqrt(a.var()/len(a)+b.var()/len(b)),
                                p_value=stats.ttest_ind(a,b,axis=0,equal_var=False,nan_policy='omit').pvalue))
            z.index.name='gene_symbol'
            save(z.reset_index(),'输入/GSE185909_原表达尺度Welch差异.csv.gz')
        z=z.replace([np.inf,-np.inf],np.nan).dropna(subset=['logFC','se','p_value']);z=z[z.se>0]
        n1=int(tm.y.sum());n0=int((1-tm.y).sum());df=n1+n0-2
        correction=np.exp(gammaln(df/2)-.5*np.log(df/2)-gammaln((df-1)/2))
        z['g']=correction*z.logFC/z.se*np.sqrt(1/n1+1/n0)
        z=z.rename_axis('gene_symbol').reset_index().assign(region=r,geo=g,unit=uid,family=family(g))
        frames.append(z)
        roster.append(dict(unit=uid,region=r,geo=g,family=family(g),profiles=len(md),donor_tissue_records=len(tm),AD=n1,control=n0))
    long=pd.concat(frames,ignore_index=True);save(long,'输入/规范基因队列差异与模型衍生g.csv.gz');save(roster,'输入/最新全发现队列清单.csv')
    results={}; familydata={}; programs=[]; go_results=[]
    go, names=pw.load_go_bp()
    # Resolve GO aliases with the same approved-gene resolver used in current panels.
    resolver=old.base.idmap.Names()
    allsyms=set.union(*go.values()); canon={s:resolver.resolve([s],[])[0] for s in allsyms}
    go={t:{canon[s] for s in gs if canon.get(s)} for t,gs in go.items()}
    save([dict(go_id=t,gene_symbol=g) for t,gs in go.items() for g in sorted(gs)],'输入/规范GO_BP成员.csv.gz')
    for i,scope in enumerate(['whole_brain',*sorted(long.region.unique())]):
        data=long if scope=='whole_brain' else long[long.region.eq(scope)]
        m,d=integrate(data); results[scope]=m; familydata[scope]=d
        blocks={f:v.set_index('gene_symbol')[['logFC','se','p_value']] for f,v in d.groupby('family')}
        q=old.qval(blocks,0 if scope=='whole_brain' else 10*old.sr.REGION_ORDER.index(scope))
        m['SumRank_FDR']=m.gene_symbol.map(q)
        save(m,'结果/'+scope+'_基因Meta.csv.gz');save(d,'输入/'+scope+'_家族级差异.csv.gz')
        gp=pw.analyze_ranking(m.rename(columns={'effect':'logFC','p':'p_value'}),go,names).assign(scope=scope)
        go_results.append(gp)
        axis=pw.analyze_ranking(m.rename(columns={'effect':'logFC','p':'p_value'}),memberships,{a:a for a in memberships},min_size=10,max_size=50000).assign(scope=scope)
        programs.append(axis)
        print('META',scope,'families',d.family.nunique(),'genes',len(m),'FDR<.05',int(m.fdr.lt(.05).sum()),flush=True)
    save(pd.concat(go_results),'结果/GO_BP方向性结果.csv.gz');save(pd.concat(programs),'结果/六机制方向性秩结果.csv')
    summaries=[]; paired=[]; baseline=results['whole_brain']
    for r,m in results.items():
        if r=='whole_brain':continue
        z=m.merge(baseline,on='gene_symbol',suffixes=('_region','_whole'))
        z=z[(z.k_region>=2)&(z.k_whole>=2)].copy()
        z['I2_change']=z.I2_region-z.I2_whole
        z['abs_g_change']=abs(z.median_g_star_region)-abs(z.median_g_star_whole)
        paired.append(z.assign(region=r))
        summaries.append(dict(region=r,common_genes=len(z),median_I2_region=z.I2_region.median(),median_I2_whole=z.I2_whole.median(),
            paired_I2_change=z.I2_change.median(),fraction_I2_lower=(z.I2_change<0).mean(),
            median_abs_g_region=abs(z.median_g_star_region).median(),median_abs_g_whole=abs(z.median_g_star_whole).median(),
            paired_abs_g_change=z.abs_g_change.median()))
    save(summaries,'结果/分脑区效应量与I2变化.csv');save(pd.concat(paired),'结果/逐基因分层前后配对.csv.gz')
    common=set.intersection(*(set(m.gene_symbol) for r,m in results.items() if r!='whole_brain'))
    sig=pd.DataFrame({r:m.set_index('gene_symbol').fdr.reindex(sorted(common))<.05 for r,m in results.items() if r!='whole_brain'})
    sig['significant_regions']=sig.sum(axis=1);save(sig.rename_axis('gene_symbol').reset_index(),'结果/六脑区共同背景显著性.csv.gz')
    save([dict(at_least_regions=n,genes=int((sig.significant_regions>=n).sum())) for n in range(1,7)],'结果/显著脑区数量分布.csv')
    # Current mechanisms on canonicalized direct interaction statistics. Model estimates are not invented anew.
    direct=[]
    for fam in ['GSE118553','GSE48350','GSE36980','HBTRC']:
        p=BASE/'region_disease_interaction_paired'/(fam+'_pairwise_gene_interactions.csv');FILES.add(p)
        v=pd.read_csv(p)
        v['original_gene_symbol']=v.gene_symbol
        v['gene_symbol']=[resolver.resolve([s],[])[0] for s in v.gene_symbol]
        v=v[v.gene_symbol.ne('')].sort_values('p_value').drop_duplicates(['pair','gene_symbol'])
        for pair,idx in v.groupby('pair').groups.items():v.loc[idx,'fdr']=meta.bh_adjust(v.loc[idx,'p_value'])
        direct.append(v.assign(family=fam))
    save(pd.concat(direct),'交互结果/原模型规范基因交互.csv.gz')
    for g in ['GSE118553','GSE48350','GSE36980','GSE122063']:
        for uid in units:
            if uid.endswith('__'+g):raw_unit(uid,manifest)
    # Archive source fingerprints; later stages append their own inputs separately.
    save([dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(FILES)],'核对/输入SHA256.csv')
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

def resume_exports():
    manifest=pd.read_csv(BASE/'audit/fixed_region_input_manifest.csv')
    units=frozen_units(read(DB/'可测.pkl').columns)
    for p in (OUT/'输入').glob('*_家族级差异.csv.gz'):
        scope=p.name.split('_家族级差异')[0]
        d=pd.read_csv(p,float_precision='round_trip')
        blocks={f:v.set_index('gene_symbol')[['logFC','se','p_value']] for f,v in d.groupby('family')}
        q=old.qval(blocks,0 if scope=='whole_brain' else 10*old.sr.REGION_ORDER.index(scope))
        path=OUT/'结果'/(scope+'_基因Meta.csv.gz');m=pd.read_csv(path,float_precision='round_trip')
        m['SumRank_FDR']=m.gene_symbol.map(q);m.to_csv(path,index=False,encoding='utf-8-sig',float_format='%.17g')
    for g in ['GSE118553','GSE48350','GSE36980','GSE122063']:
        for uid in units:
            if uid.endswith('__'+g):raw_unit(uid,manifest);print('Exported raw interaction',uid,flush=True)
    for uid in units:
        FILES.update(DB/(uid+'_'+suffix+'.pkl') for suffix in ['de','info','md'])
    FILES.update([old.CACHE/'机制归属.csv.gz',BASE/'audit/fixed_region_input_manifest.csv',pw.GO_DB,pw.ORG_DB])
    save([dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(FILES)],'核对/输入SHA256.csv')

if __name__=='__main__':
    with threadpool_limits(limits=1):
        if '--resume-exports' in sys.argv:resume_exports()
        else:main()
