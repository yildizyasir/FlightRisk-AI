# Offline-first route map: accurate Natural Earth outline, no external tile provider.
from __future__ import annotations
import html
import math
from pathlib import Path
from inference import airport_details

W,H=950,490
XMIN,XMAX,YMIN,YMAX=-128,-65,22,51
BASE_SVG=Path(__file__).with_name('route_map_base.svg').read_text(encoding='utf-8')

def xy(lon,lat):
    return ((float(lon)-XMIN)/(XMAX-XMIN)*W,(YMAX-float(lat))/(YMAX-YMIN)*H)

def route_map_html(origin,dest,meta):
    a=airport_details(origin,meta)
    b=airport_details(dest,meta)
    if any(a[k] is None or b[k] is None for k in ('lat','lon')):
        return '<div style="color:white;padding:30px">Map not available for this airport pair.</div>'
    x1,y1=xy(a['lon'],a['lat'])
    x2,y2=xy(b['lon'],b['lat'])
    cx,cy=(x1+x2)/2,(y1+y2)/2-max(36,min(115,abs(x1-x2)*.18))
    # Use dynamic SVG paths; no JavaScript or external CDN dependencies.
    route=f'''<path d="M{x1:.1f},{y1:.1f} Q{cx:.1f},{cy:.1f} {x2:.1f},{y2:.1f}" fill="none" stroke="#75eed4" stroke-width="4" stroke-linecap="round" stroke-dasharray="11 8"/>
    <circle cx="{x1:.1f}" cy="{y1:.1f}" r="8" fill="#75eed4" stroke="#0b2536" stroke-width="5"/>
    <circle cx="{x2:.1f}" cy="{y2:.1f}" r="8" fill="#75eed4" stroke="#0b2536" stroke-width="5"/>
    <text x="{x1+12:.1f}" y="{y1-15:.1f}" font-size="22" fill="#effffb" font-family="sans-serif" font-weight="bold">{html.escape(origin)}</text>
    <text x="{x2+12:.1f}" y="{y2-15:.1f}" font-size="22" fill="#effffb" font-family="sans-serif" font-weight="bold">{html.escape(dest)}</text>'''
    svg=BASE_SVG.replace('<g id="route-line-layer">', '<g id="route-line-layer">',1)
    start=svg.index('<g id="route-line-layer">')
    end=svg.index('</g>',start)+len('</g>')
    svg=svg[:start]+'<g id="route-line-layer">'+route+'</g>'+svg[end:]
    return f'''<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"/>
    <style>*{{box-sizing:border-box}}body{{margin:0;background:#0b2131;color:#effafb;font-family:system-ui,sans-serif}}
    .wrap{{height:100vh;position:relative;overflow:hidden;border:1px solid #294b5b;border-radius:16px;background:#0c2638}}
    svg{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}}
    .title{{position:absolute;left:23px;top:21px;font-size:17px;font-weight:800;background:#0b2032be;border-radius:10px;padding:10px 12px}}
    .sub{{display:block;color:#aac5ce;font-size:11px;font-weight:500;margin-top:5px}}
    .legend{{position:absolute;right:20px;top:23px;font-size:10px;letter-spacing:1.6px;border:1px solid #457268;background:#0a292eb8;color:#8ff6dd;border-radius:9px;padding:10px 12px}}
    .footer{{position:absolute;bottom:17px;left:23px;right:23px;background:#0a2233e8;color:#f1fbfc;border:1px solid #34556b;border-radius:11px;padding:12px 15px;display:flex;justify-content:space-between;align-items:center;font-size:12px;gap:15px}}
    .footer strong{{font-size:16px;font-weight:800}}.footer span{{color:#b0cdd7;font-size:11px}}
    </style></head><body><div class="wrap">{svg}
    <div class="title">Your flight route<span class="sub">Geographical route overview · approximate arc</span></div>
    <div class="legend">ROUTE MAP</div>
    <div class="footer"><div><strong>{html.escape(origin)}</strong><br><span>{html.escape(a['city'])}</span></div><div style="color:#75eed4">✈</div><div style="text-align:right"><strong>{html.escape(dest)}</strong><br><span>{html.escape(b['city'])}</span></div></div>
    </div></body></html>'''
