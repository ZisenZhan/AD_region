"""Compact display of the unchanged November 13 matrices."""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

out = Path(__file__).resolve().parent
root = out.parents[1]
spec = importlib.util.spec_from_file_location("original", root / "脚本/绘制热力图.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
matrices = []
for cn, title in old.REGIONS.values():
    df = pd.read_csv(old.SOURCE / (cn + "_六机制底座A.csv"), index_col="model", float_precision="round_trip")
    assert df.index.tolist() == list(old.models.model) and df.columns[-2] == "Dev macro"
    matrices.append((cn, title, df))


def draw(selected, name):
    cell = .247
    left, gap, right, bottom, field, height = 1.68, .23, .10, 1.10, 4.62, 5.99
    width = left + sum(cell * t[2].shape[1] for t in selected) + gap * (len(selected)-1) + right
    fig = plt.figure(figsize=(width,height), dpi=140)
    pos = left
    for i, (_, title, df) in enumerate(selected):
        n = df.shape[1]
        ax = fig.add_axes([pos/width, bottom/height, n*cell/width, field/height])
        vals = df.to_numpy()
        ax.imshow(np.ma.masked_invalid(vals), aspect="auto", cmap=old.CMAP,norm=old.NORM,interpolation="none")
        ax.set_xticks(range(n), list(df.columns[:-2])+["Dev macro",df.columns[-1].split()[-1]],
                      rotation=90,ha="center",va="top",fontsize=6)
        ax.tick_params(axis="x",labelbottom=True,labeltop=False)
        ax.set_yticks(range(36), old.LABELS if i==0 else [""]*36,fontsize=5.8)
        ax.tick_params(length=0,pad=2.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(.5,n-.5,1),minor=True)
        ax.set_yticks(np.arange(.5,35.5,1),minor=True)
        ax.tick_params(which="minor",length=0)
        ax.grid(which="minor",color="white",linewidth=.28)
        for b in old.BOUNDARIES:
            ax.axhline(b-.5,color="#8c979e",linewidth=.43)
        for j in [n-2,n-1]:
            ax.axvline(j-.5,color="#667681",lw=.7)
        rr = df.index.get_loc("M23")
        ax.add_patch(Rectangle((-.5,rr-.5),n,1,fill=False,ec="#263e4c",lw=.8,clip_on=False,zorder=6))
        if i==0:
            ax.get_yticklabels()[rr].set_weight("bold")
        txts=[]
        for (r,c),v in np.ndenumerate(vals):
            txts.append(ax.text(c,r,f"{v:.3f}" if np.isfinite(v) else "NA",ha="center",va="center",
                       fontsize=5.6,color="white" if v>.9 or v<.16 else "#192329"))
        top = bottom+field
        ax.plot([-.5,n-1.53],[-.55,-.55],lw=2,color="#83969f",clip_on=False)
        ax.plot([n-1.49,n-.51],[-.55,-.55],lw=2,color="#176d76",clip_on=False)
        ax.set_ylim(35.5,-.5)
        ax.set_xlim(-.5,n-.5)
        ax.get_xticklabels()[-1].set_color("#176d76")
        ax.get_xticklabels()[-1].set_weight("bold")
        fig.text((pos+n*cell/2)/width,.39/height,title,ha="center",fontsize=8,weight="bold")
        fig.text((pos+(n-1)*cell/2)/width,(top+.065)/height,"Development set",ha="center",fontsize=6.2,color="#52636b")
        fig.text((pos+(n-.5)*cell)/width,(top+.065)/height,"External set",ha="center",fontsize=6.2,color="#176d76")
        fig.canvas.draw()
        renderer=fig.canvas.get_renderer()
        assert all(t.get_window_extent(renderer).width < cell*fig.dpi for t in txts)
        pos+=n*cell+gap
    cbax=fig.add_axes([(width-right-1.18)/width,.089/height,1.18/width,.058/height])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=old.NORM,cmap=old.CMAP),cax=cbax,
                    orientation="horizontal",ticks=[0,.5,1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1,pad=1,labelsize=5.6)
    fig.text((width-right-1.26)/width,.117/height,"AUC",ha="right",va="center",fontsize=6)
    for ext in ("png","svg","pdf"):
        fig.savefig(out/(name+"."+ext),dpi=400,bbox_inches="tight",pad_inches=.045)
    plt.close(fig)


draw(matrices,"三脑区_36模型紧凑热图")
for m in matrices:
    draw([m],m[0]+"_36模型紧凑热图")
