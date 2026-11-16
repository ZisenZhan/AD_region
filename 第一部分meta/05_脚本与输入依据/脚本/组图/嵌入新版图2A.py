"""Place the approved editable PDF/SVG into panel A without rasterizing the figure."""
import io
import re
import shutil
from pathlib import Path
from lxml import etree as ET
import fitz

def export_current(f, panels, positions, o):
    source=o.parent/'图2A_经典Hedges_g/前100紧凑版'
    asset=source/'图2A_前100_仅基因聚类'
    name='图02_区域机制模式与分层收益'
    target=o/'主文组图'/name
    for i,(x,y) in enumerate(positions):
        f.text(x,y,chr(65+i),fontsize=11,ha='left',va='top')
    bg=io.BytesIO()
    f.savefig(bg,format='pdf')
    with fitz.open(stream=bg.getvalue(),filetype='pdf') as doc,fitz.open(asset.with_suffix('.pdf')) as panel:
        page=doc[0]; w,h=page.rect.width,page.rect.height
        box=fitz.Rect(.012*w,.056*h,.577*w,.758*h)
        scale=min(box.width/panel[0].rect.width,box.height/panel[0].rect.height)
        sw,sh=panel[0].rect.width*scale,panel[0].rect.height*scale
        left,top=box.x0+(box.width-sw)/2,box.y0+(box.height-sh)/2
        page.show_pdf_page(fitz.Rect(left,top,left+sw,top+sh),panel,0)
        doc.save(target.with_suffix('.pdf'),garbage=4,deflate=True)
        page.get_pixmap(dpi=420).save(target.with_suffix('.png'))
    svg=io.BytesIO()
    f.savefig(svg,format='svg')
    root=ET.fromstring(svg.getvalue())
    art=ET.parse(str(asset.with_suffix('.svg'))).getroot()
    ids={v:'top100_'+v for v in art.xpath('//@id')}
    for node in art.iter():
        for key,value in list(node.attrib.items()):
            if key=='id': node.set(key,ids[value])
            elif value.startswith('#') and value[1:] in ids: node.set(key,'#'+ids[value[1:]])
            elif 'url(#' in value: node.set(key,re.sub(r'url\(#([^)]*)\)',lambda m:'url(#'+ids[m[1]]+')',value))
    art.set('x',str(left)); art.set('y',str(top)); art.set('width',str(sw)); art.set('height',str(sh))
    root.append(art)
    ET.ElementTree(root).write(str(target.with_suffix('.svg')),encoding='utf-8',xml_declaration=True)
    stem=o/'单独子图'/(name+'_A_跨队列与区域Meta热图')
    for ext in ['.pdf','.png','.svg']:
        shutil.copyfile(asset.with_suffix(ext),stem.with_suffix(ext))
    for fn in ['前100基因及六脑区结果.csv','热图逐格数据.csv']:
        shutil.copyfile(source/fn,o/'源数据'/('图2A_经典g_'+fn))
