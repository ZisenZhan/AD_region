from pathlib import Path
import sys
import runpy
import numpy as np
import pandas as pd
from scipy import stats, special

O = Path(__file__).resolve().parent
B = O.parent
S = B / '第一部分_最新队列六机制重算_20261113'
A = B / '别名审计与评分轮换_1107'
F = B / '整脑_区域机制_额叶星形胶质_紧凑组图_20261116'
D = O / '源数据'
D.mkdir(exist_ok=True)
def save(x, name):
    x.to_csv(D/name, index=False, encoding='utf-8-sig')

roster = pd.read_csv(S/'输入/最新全发现队列清单.csv')
manifest = pd.read_csv(B/'audit/fixed_region_input_manifest.csv')
oldtests = pd.read_csv(S/'输入/规范基因队列差异与模型衍生g.csv.gz',usecols=['unit','gene_symbol'])
cols = pd.read_csv(F/'源数据/热图列信息.csv')
old = pd.read_csv(F/'源数据/Top50预定规则名单.csv')

def expression(row,md):
    if row.geo in ['GSE122063','GSE37263']:
        x = pd.read_csv(A/'规范输入'/(row.unit+'_expression.tsv.gz'),sep='\t',index_col=0).T
        x.index=x.index.astype(str)
        return x.loc[md.key.astype(str)].set_axis(md.index)
    parts=[]
    for m in manifest[manifest.unit_id.eq(row.unit)].itertuples():
        sep='\t' if '.tsv' in m.matrix_path else ','
        hdr=pd.read_csv(m.matrix_path,sep=sep,nrows=0).columns
        labels=md.source_sample_ids.map(lambda s: [g for g in s.split(';') if g in hdr])
        assert labels.map(len).eq(1).all(),row.unit
        ids=labels.str[0].tolist()
        raw=pd.read_csv(m.matrix_path,sep=sep,usecols=[hdr[0],*ids],dtype={hdr[0]:str}).set_index(hdr[0])[ids]
        if m.assay_class=='rnaseq_counts':
            assert raw.sum().gt(0).all()
            raw=np.log2(raw.div(raw.sum())*1e6+.5)
        raw.index=raw.index.str.strip().str.replace(r'\.0$','',regex=True)
        mp=pd.read_csv(A/'审计'/(row.geo+'__'+m.component+'_全部特征.csv.gz'),dtype=str,keep_default_na=False).set_index('feature_id').gene_symbol
        gs=mp.reindex(raw.index)
        assert gs.notna().all()
        x=raw.loc[gs.ne('').values].groupby(gs[gs.ne('')].values).median().T
        parts.append(x.set_axis(md.index))
    assert parts
    return parts[0] if len(parts)==1 else pd.concat(parts,axis=1).T.groupby(level=0).median().T

def compute(row):
    md=pd.read_pickle(A/'训练与评测缓存'/(row.unit+'_md.pkl'))
    assert md.groupby('key').y.nunique().eq(1).all()
    x=expression(row,md).groupby(md.key).mean()
    y=md.groupby('key').y.first().reindex(x.index)
    assert (sum(y==1),sum(y==0))==(row.AD,row.control)
    x=x.reindex(columns=oldtests.loc[oldtests.unit.eq(row.unit),'gene_symbol'])
    a,b=x.loc[y.eq(1)],x.loc[y.eq(0)]
    n,m=a.count(),b.count()
    df=n+m-2
    sp=np.sqrt(((n-1)*a.var()+(m-1)*b.var())/df)
    z=pd.DataFrame(dict(n_AD=n,n_control=m,mean_AD=a.mean(),mean_control=b.mean(),pooled_SD=sp))
    z=z[(n>=2)&(m>=2)&sp.gt(0)&np.isfinite(sp)].copy()
    n,m=z.n_AD,z.n_control
    df=n+m-2
    d=(z.mean_AD-z.mean_control)/z.pooled_SD
    j=np.exp(special.gammaln(df/2)-.5*np.log(df/2)-special.gammaln((df-1)/2))
    z['effect']=j*d
    z['variance']=(n+m)/(n*m)+z.effect**2/(2*(n+m))
    z['p']=2*stats.t.sf(abs(d)/np.sqrt(1/n+1/m),df)
    z['fdr']=stats.false_discovery_control(z.p)
    return z.rename_axis('gene_symbol').reset_index().assign(unit=row.unit,region=row.region,family=row.family)

def meta(v):
    # Repeated contributions within a source family do not gain precision.
    w=v.groupby(['family','gene_symbol'])[['effect','variance']].mean().reset_index()
    e=w.pivot(index='gene_symbol',columns='family',values='effect')
    iv=1/w.pivot(index='gene_symbol',columns='family',values='variance')
    k=e.count(axis=1)
    s=iv.sum(axis=1)
    fixed=(iv*e).sum(axis=1)/s
    q=(iv*e.sub(fixed,axis=0)**2).sum(axis=1)
    c=s-(iv**2).sum(axis=1)/s
    tau2=((q-k+1)/c.where(k>1)).fillna(0).clip(lower=0)
    rw=1/(1/iv).add(tau2,axis=0)
    effect=(rw*e).sum(axis=1)/rw.sum(axis=1)
    se=1/np.sqrt(rw.sum(axis=1))
    p=2*stats.norm.sf(abs(effect/se))
    return pd.DataFrame(dict(effect=effect,se=se,tau2=tau2,k=k,p=p,fdr=stats.false_discovery_control(p))).reset_index()

def main():
    tables=[]
    for row in roster.itertuples():
        print(row.unit,flush=True)
        tables.append(compute(row))
    long=pd.concat(tables,ignore_index=True)
    save(long,'经典g_队列全基因.csv.gz')
    mt=pd.concat([meta(v).assign(region=r,unit=r) for r,v in long.groupby('region')],ignore_index=True)
    save(mt,'经典g_六脑区Meta全基因.csv.gz')
    g=mt.pivot(index='gene_symbol',columns='region',values='effect').dropna()
    q=mt.pivot(index='gene_symbol',columns='region',values='fdr').loc[g.index]
    rank=g.add_prefix('g_').join(q.add_prefix('fdr_')).assign(significant_regions=(q<.05).sum(axis=1),mean_abs_g=g.abs().mean(axis=1)).reset_index()
    pool=rank.query('significant_regions>=4').sort_values(['mean_abs_g','gene_symbol'],ascending=[False,True])
    top=pool.head(50)
    assert len(top)==50
    save(pool,'至少四区显著_全部候选.csv')
    save(top,'新版前50.csv')
    both=old[['gene_symbol']].assign(old_rank=range(1,51)).merge(top[['gene_symbol']].assign(new_rank=range(1,51)),how='outer',indicator='membership')
    save(both,'两版前50逐基因对照.csv')
    bothn=both.membership.eq('both').sum()
    print('COMMON',len(g),'POOL',len(pool),'OVERLAP',bothn,flush=True)
    src=pd.concat([long,mt],ignore_index=True).set_index(['unit','gene_symbol'])
    heat=cols.merge(top[['gene_symbol']],how='cross').join(src[['effect','fdr']],on=['unit','gene_symbol'])
    render(heat,top)

def render(heat,top):
    sys.dont_write_bytecode=True
    style=runpy.run_path(str(F/'脚本/绘制紧凑组图.py'))
    draw=style['cohort_heat']
    draw.__globals__.update(D=D,heat=heat,top=top,colinfo=cols)
    fig=style['newfig']((7.2,6.2))
    ax=fig.add_axes([.28,.21,.62,.65])
    draw(ax)
    im=ax.images[0]
    with im.callbacks.blocked(signal='changed'):
        im.set_clim(-2,2)
    ca=ax.child_axes[-1]
    ca.clear()
    cb=fig.colorbar(im,cax=ca,orientation='horizontal',ticks=[-2,-1,0,1,2],extend='both')
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=5,length=1.5)
    cb.set_label("Hedges' g",fontsize=5.8,labelpad=1.2)
    for ext in ['png','svg','pdf']:
        fig.savefig(O/('图2A_经典Hedges_g.'+ext),dpi=360,bbox_inches='tight',pad_inches=.07)
    style['plt'].close(fig)

if __name__=='__main__':
    if '--render-only' in sys.argv:
        h=pd.read_csv(D/'图2A_全部队列及Meta显著性黑点.csv')
        render(h.drop(columns=['significance_dot','display_row','display_column']),pd.read_csv(D/'新版前50.csv'))
    else:
        main()
