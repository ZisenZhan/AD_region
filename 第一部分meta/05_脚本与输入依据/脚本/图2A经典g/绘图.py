from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import pdist
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

O=Path(__file__).resolve().parent
D=O.parent/'源数据'
old=O.parent.parent/'整脑_区域机制_额叶星形胶质_紧凑组图_20261116/源数据'
ci=pd.read_csv(old/'热图列信息.csv')
cohort=pd.read_csv(D/'经典g_队列全基因.csv.gz',usecols=['gene_symbol','unit','effect','fdr'])
meta=pd.read_csv(D/'经典g_六脑区Meta全基因.csv.gz',usecols=['gene_symbol','unit','effect','fdr'])
pool=pd.read_csv(D/'至少四区显著_全部候选.csv')
top=pool.sort_values(['mean_abs_g','gene_symbol'],ascending=[False,True]).head(100).copy()
assert len(top)==top.gene_symbol.nunique()==100 and top.significant_regions.min()>=4
assert top.gene_symbol.iloc[:50].tolist()==pd.read_csv(D/'新版前50.csv').gene_symbol.tolist()
top.insert(0,'rank',range(1,101))
def save(x,name):
    x.to_csv(O/name,encoding='utf-8-sig',index=False)
save(top,'前100基因及六脑区结果.csv')
src=pd.concat([cohort,meta[['gene_symbol','unit','effect','fdr']]])
t=ci.merge(top[['gene_symbol','rank']],how='cross').merge(src,on=['unit','gene_symbol'],how='left',validate='one_to_one')
def matrix(value):
    return t.pivot(index='gene_symbol',columns='unit',values=value).loc[top.gene_symbol,ci.unit]
h=matrix('effect')
z=h.loc[:,ci.loc[~ci.is_meta,'unit']].rank(pct=True)
def distance(a,b):
    keep=np.isfinite(a)&np.isfinite(b)
    assert keep.sum()>=2
    return np.sqrt(np.mean((a[keep]-b[keep])**2))
def cluster(x):
    return linkage(pdist(x,metric=distance),method='average',optimal_ordering=True)
tree=cluster(z.to_numpy())
rows=h.index[leaves_list(tree)]
cols=[]; groups=[]
for r,c in ci.groupby('region',sort=False):
    units=c.loc[~c.is_meta,'unit'].to_numpy()
    start=len(cols)
    cols.extend(units)
    cols.append(c.loc[c.is_meta,'unit'].item())
    groups.append((r,start,len(cols)))
assert cols==ci.unit.tolist()
ci=ci.set_index('unit').loc[cols]
width=np.where(ci.is_meta,1.85,1.)
x=np.r_[0.,np.cumsum(width)]; mid=(x[1:]+x[:-1])/2
eff=h.loc[rows,cols]
q=t.pivot(index='gene_symbol',columns='unit',values='fdr').loc[rows,cols]
t['display_row']=t.gene_symbol.map(dict(zip(rows,range(1,101))))
t['display_col']=t.unit.map(dict(zip(cols,range(1,35))))
save(t.sort_values(['display_row','display_col']),'热图逐格数据.csv')
mpl.rcParams.update({'font.family':'Arial','pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.4})
fig=plt.figure(figsize=(6.05,8.85))
left,bottom,w,height=.178,.025,.743,.847
ax=fig.add_axes([left,bottom,w,height])
cmap=LinearSegmentedColormap.from_list('effect',['#3826a9','#aa9ce2','#ffffff','#f4ada3','#cf292f'])
cmap.set_bad('#e2e2e2')
mesh=ax.pcolormesh(x,np.arange(101),np.ma.masked_invalid(eff),cmap=cmap,vmin=-2,vmax=2,edgecolors='none',antialiased=False)
ax.set(xlim=(0,x[-1]),ylim=(100,0))
ax.set_yticks(np.arange(100)+.5,rows,fontsize=4.6)
ax.set_xticks(mid,['Meta' if m else label for m,label in zip(ci.is_meta,ci.label)],fontsize=4.8,rotation=90)
ax.xaxis.tick_top()
ax.tick_params(which='both',length=0,pad=1.1)
for s in ax.spines.values(): s.set_visible(False)
yi,xi=np.where(q.to_numpy()<.05)
ax.scatter(mid[xi],yi+.5,s=.70,c='#221c2a',linewidths=0)
for j in np.flatnonzero(ci.is_meta):
    ax.add_patch(Rectangle((x[j],0),width[j],100,fill=False,edgecolor='#46404b',lw=.48,clip_on=False))
def draw_tree(axis,tree,orientation):
    dendrogram(tree,ax=axis,orientation=orientation,no_labels=True,color_threshold=0,above_threshold_color='#857e89')
    for l in axis.collections:l.set_linewidth(.38)
    axis.axis('off')
rowax=fig.add_axes([.012,bottom,.076,height])
draw_tree(rowax,tree,'left')
rowax.set_ylim(1000,0)
names=['Cerebellum','Occipital','Frontal','Temporal','Hippocampus','Entorhinal']
colors=['#00A88F','#4CB6E0','#7A4BE1','#EF3D94','#F3AA32','#F04D47']
for (r,a,b),name,color in zip(groups,names,colors):
    def xx(i):return left+w*x[i]/x[-1]
    fig.add_artist(plt.Line2D([xx(a)+.001,xx(b)-.001],[.919,.919],transform=fig.transFigure,color=color,lw=1.6,solid_capstyle='butt'))
    fig.text((xx(a)+xx(b))/2,.924,name,ha='center',va='bottom',fontsize=5.6)
cax=fig.add_axes([.942,.41,.013,.145])
cb=fig.colorbar(mesh,cax=cax,ticks=[-2,0,2],extend='both')
cb.outline.set_visible(False); cb.ax.tick_params(length=1.5,pad=1,labelsize=5)
cb.ax.set_title("Hedges’ g",fontsize=5.5,pad=4)
for ext in ['png','pdf','svg']:
    fig.savefig(O/('图2A_前100_仅基因聚类.'+ext),dpi=400,bbox_inches='tight',pad_inches=.06)
plt.close(fig)
print('genes, columns, significant cells:',*eff.shape,len(xi))
