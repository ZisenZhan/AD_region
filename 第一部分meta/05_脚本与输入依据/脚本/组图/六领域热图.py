from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,leaves_list
from matplotlib.patches import Rectangle

O=Path(__file__).resolve().parents[1]
N=O/'六领域GO与额叶单核'
AX=['Synapse','Mitochondrial Metabolism','Immune Response','Structural Stabilization','Proteostasis','Lipid Metabolism']
COL=['#345da9','#00887c','#c04c44','#9868ac','#b17725','#558340']
def short(s):
    names={'regulation of microtubule cytoskeleton organization':'Microtubule organization regulation',
    'endoplasmic reticulum tubular network organization':'ER tubular network organization',
    'positive regulation of synaptic transmission':'Positive reg. of synaptic transmission',
    'response to lipopolysaccharide':'Response to lipopolysaccharide',
    'vesicle-mediated transport to the plasma membrane':'Vesicle transport to plasma membrane',
    'cellular response to steroid hormone stimulus':'Cellular response to steroid hormone',
    'positive regulation of cholesterol efflux':'Positive reg. of cholesterol efflux'}
    return names.get(s,s.replace('regulation of ','reg. of ').replace('extracellular matrix','ECM').replace('biosynthetic process','biosynthesis').replace('metabolic process','metabolism')).capitalize().replace('ecm','ECM').replace('Er tubular','ER tubular').replace('Atp ','ATP ').replace('Rrna ','rRNA ')
def plot(ax,raw,regions,rnames,types,ctnames,cmap,mode):
    spec=pd.read_csv(N/'图2C冻结清单.csv').set_index('go_id')
    src=raw[raw.scope.isin(regions)&raw.go_id.isin(spec.index)]
    z=src.pivot(index='go_id',columns='scope',values='directional_rank_z').loc[spec.index,regions]
    q=src.pivot(index='go_id',columns='scope',values='directional_rank_fdr').loc[spec.index,regions]
    zz=z.sub(z.mean(axis=1),axis=0).div(z.std(axis=1),axis=0)
    order=z.index[leaves_list(linkage(zz,method='average',optimal_ordering=True))]
    if mode=='bulk':
        val=z.loc[order];sig=q.loc[order]<.05;cols=rnames;limit=12;cblabel='Signed rank Z'
    else:
        order=order[q.loc[order,'frontal_prefrontal'].values<.05]
        tests=pd.read_csv(N/'八类细胞完整检验.csv')
        def mat(col):return tests.pivot(index='go_id',columns='cell_type',values=col).loc[order,types]
        val=-np.log10(mat('p').clip(lower=1e-20))*np.where(mat('direction')=='Up',1,-1)
        sig=mat('fdr')<.05;cols=ctnames;limit=4;cblabel='Signed -log10(P)'
    xp=np.arange(len(cols));yp=np.arange(len(order))
    im=ax.imshow(val,cmap=cmap,vmin=-limit,vmax=limit,aspect='auto',interpolation='nearest')
    ax.set_xticks(xp,cols,rotation=60,ha='right',fontsize=5.1)
    ax.set_yticks(yp,[short(spec.at[g,'term']) for g in order],fontsize=4.8)
    for s in ax.spines.values():s.set_visible(False)
    ax.tick_params(axis='both',length=0,pad=1)
    # One continuous strip, aligned to the heatmap's row edges.
    pts=ax.get_position().width*ax.figure.get_figwidth()*72
    ax.tick_params(axis='y',pad=8)
    for i,g in enumerate(order):
        domains=spec.at[g,'domains'].split(';')
        width=4.8/len(domains)
        for j,domain in enumerate(domains):
            ax.add_patch(Rectangle(((-6.3+j*width)/pts,i-.5),width/pts,1,
                transform=ax.get_yaxis_transform(),facecolor=COL[AX.index(domain)],
                edgecolor='none',linewidth=0,antialiased=False,clip_on=False))
    key=ax.inset_axes([-1.72,1.02,2.68,.088] if mode=='bulk' else [-129/pts,1.012,123/pts,.085])
    key.axis('off')
    for j,n in enumerate(['Synapse','Mitochondrial','Immune','Structural','Proteostasis','Lipid']):
        xx=(j%3)/3;yy=.8-.58*(j//3)
        key.plot(xx,yy,'s',color=COL[j],mec='none',ms=3,transform=key.transAxes,clip_on=False)
        key.text(xx+.033,yy,n,va='center',fontsize=4.7,transform=key.transAxes)
    if mode!='bulk':
        ax.add_patch(Rectangle((types.index('Astrocyte')-.5,-.5),1,len(order),fc='none',ec='#202129',lw=.8))
        ax.set_title('Frontal bulk-pattern correspondence',fontsize=7,loc='left',fontweight='bold',pad=4)
    for i,j in zip(*np.where(sig)):
        ax.plot(j,i,'o',color='#24212c' if abs(val.iloc[i,j])<limit*.70 else 'white',ms=1.65,mec='none')
    ca=ax.inset_axes([1.04,.18,.037,.60])
    cb=ax.figure.colorbar(im,cax=ca,ticks=[-limit,0,limit]);cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.3,labelsize=4.7,pad=1);cb.set_label(cblabel,fontsize=5,labelpad=1.5)
    pd.DataFrame({'go_id':order,'label':[short(spec.at[g,'term']) for g in order]}).to_csv(N/('图2C显示顺序.csv' if mode=='bulk' else '图3A显示顺序.csv'),index=False,encoding='utf-8-sig')
