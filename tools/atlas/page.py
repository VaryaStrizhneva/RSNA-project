"""One self-contained HTML page per pathology.

Self-contained on purpose: every slice is a data URI inside the file, so the page opens
from disk with no server and nothing to fetch. That also keeps it off the network,
which matters — these are real patient studies, de-identified but not ours to publish.

The viewer is a slider rather than a contact sheet because that is how the images are
actually read: a tear appears over two or three slices and vanishes again, and that
*persistence across slices* is most of the signal. A grid of thumbnails throws it away.
"""

from __future__ import annotations

import html
import json


CSS = """
:root{--bg:#0e1012;--panel:#16191d;--line:#252a30;--ink:#e6e8ea;--dim:#9aa3ad;
      --accent:#6ea8fe;--warn:#f0a868;--good:#7fc98b;--bad:#e0716f}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.65 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:0 24px 96px}
header{border-bottom:1px solid var(--line);padding:56px 0 28px;margin-bottom:32px}
h1{font-size:2.1rem;margin:0 0 10px;letter-spacing:-.02em}
h2{font-size:1.4rem;margin:52px 0 14px;letter-spacing:-.01em;
   border-bottom:1px solid var(--line);padding-bottom:8px}
h3{font-size:1.08rem;margin:30px 0 10px;color:var(--accent)}
.lede{color:var(--dim);font-size:1.08rem;max-width:70ch}
blockquote{border-left:3px solid var(--accent);margin:18px 0;padding:8px 18px;
           background:#12161b;color:var(--ink)}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:.94rem}
th,td{border:1px solid var(--line);padding:8px 11px;text-align:left;vertical-align:top}
th{background:#1a1e23;font-weight:600;color:var(--dim)}
table.grades td:first-child{text-align:center;font-weight:700;width:60px}
ul{padding-left:22px}li{margin:7px 0}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;
      padding:20px 22px;margin:22px 0}
.case-head{display:flex;justify-content:space-between;align-items:baseline;gap:16px;
           flex-wrap:wrap;margin-bottom:6px}
.tag{font-size:.74rem;letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;
     border-radius:99px;font-weight:700}
.tag.pos{background:#3a1f1f;color:var(--bad);border:1px solid #5b2e2c}
.tag.neg{background:#1c2f20;color:var(--good);border:1px solid #2c4a33}
.meta{color:var(--dim);font-size:.86rem;font-family:ui-monospace,monospace}
.report{background:#12161b;border:1px solid var(--line);border-radius:8px;
        padding:14px 16px;margin:14px 0;font-size:.9rem;white-space:pre-wrap;
        max-height:190px;overflow:auto;color:#c6ccd3}
.report mark{background:#4a3a12;color:#ffd98a;padding:1px 3px;border-radius:3px}
.viewer{display:grid;grid-template-columns:minmax(0,1fr) 260px;gap:20px;
        align-items:start;margin-top:16px}
@media(max-width:860px){.viewer{grid-template-columns:1fr}}
.stage{position:relative;background:#000;border-radius:8px;overflow:hidden;
       border:1px solid var(--line)}
.stage img{display:block;width:100%;height:auto}
.axis{display:flex;justify-content:space-between;color:var(--dim);font-size:.78rem;
      letter-spacing:.05em;text-transform:uppercase;margin:6px 2px 0}
.bar{position:absolute;left:14px;bottom:14px;height:4px;background:#fff;
     box-shadow:0 0 3px #000;border-radius:2px}
.barlabel{position:absolute;left:14px;bottom:22px;color:#fff;font-size:.72rem;
          text-shadow:0 0 4px #000;font-family:ui-monospace,monospace}
.flag{position:absolute;right:12px;top:12px;background:#4a3a12cc;color:#ffd98a;
      border:1px solid #6b5420;border-radius:6px;padding:4px 9px;font-size:.76rem;
      display:none}
.ctl{font-size:.86rem}
.ctl label{display:block;color:var(--dim);margin:14px 0 4px;font-size:.8rem;
           text-transform:uppercase;letter-spacing:.05em}
input[type=range]{width:100%;accent-color:var(--accent)}
.count{font-family:ui-monospace,monospace;color:var(--accent);font-size:1.15rem}
.note{color:var(--dim);font-size:.86rem;margin-top:8px}
.claude{border:1px solid #6b5420;background:#1d1a12;border-radius:8px;padding:14px 16px;
        margin-top:16px;font-size:.9rem}
.claude h4{margin:0 0 8px;color:var(--warn);font-size:.86rem;text-transform:uppercase;
           letter-spacing:.06em}
.danger{border:1px solid #6b2f2c;background:#1f1413;border-radius:8px;padding:16px 18px;
        margin:22px 0}
.danger h4{margin:0 0 8px;color:var(--bad);font-size:.86rem;text-transform:uppercase;
           letter-spacing:.06em}
.panel3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}
@media(max-width:860px){.panel3{grid-template-columns:1fr}}
.panel3 .stage img{image-rendering:auto}
.panel3 h4{margin:0 0 2px;font-size:.94rem}
.panel3 .sub{color:var(--accent);font-family:ui-monospace,monospace;font-size:.78rem}
.panel3 .why{color:var(--dim);font-size:.82rem;margin-top:4px}
footer{margin-top:80px;padding-top:20px;border-top:1px solid var(--line);
       color:var(--dim);font-size:.84rem}
kbd{background:#242a31;border:1px solid var(--line);border-radius:4px;padding:1px 6px;
    font-size:.8rem;font-family:ui-monospace,monospace}
"""

JS = """
function mount(id, frames, flagged, mmPerPxDisplayed){
  const stage=document.getElementById(id);
  const img=stage.querySelector('img');
  const slider=document.getElementById(id+'-sl');
  const count=document.getElementById(id+'-n');
  const flag=stage.querySelector('.flag');
  const bright=document.getElementById(id+'-b');
  const contrast=document.getElementById(id+'-c');
  let showFlags=false;
  function draw(){
    const i=+slider.value;
    img.src=frames[i];
    count.textContent=(i+1)+' / '+frames.length;
    flag.style.display=(showFlags&&flagged.includes(i))?'block':'none';
    img.style.filter='brightness('+bright.value+') contrast('+contrast.value+')';
  }
  slider.addEventListener('input',draw);
  bright.addEventListener('input',draw);
  contrast.addEventListener('input',draw);
  stage.tabIndex=0;
  stage.addEventListener('keydown',e=>{
    if(e.key==='ArrowRight'||e.key==='ArrowDown'){slider.value=Math.min(+slider.value+1,frames.length-1);draw();e.preventDefault();}
    if(e.key==='ArrowLeft'||e.key==='ArrowUp'){slider.value=Math.max(+slider.value-1,0);draw();e.preventDefault();}
  });
  stage.addEventListener('wheel',e=>{
    slider.value=Math.min(Math.max(+slider.value+Math.sign(e.deltaY),0),frames.length-1);
    draw();e.preventDefault();},{passive:false});
  const tog=document.getElementById(id+'-flag');
  if(tog) tog.addEventListener('change',e=>{showFlags=e.target.checked;draw();});
  // the scale bar is sized once the image knows its displayed width
  const bar=stage.querySelector('.bar'), lab=stage.querySelector('.barlabel');
  function sizeBar(){
    const px=(10/mmPerPxDisplayed)*(img.clientWidth/img.naturalWidth);
    if(isFinite(px)&&px>0){bar.style.width=px+'px';lab.textContent='10 mm';}
  }
  img.addEventListener('load',sizeBar); window.addEventListener('resize',sizeBar);
  draw();
}
"""


def _esc(s) -> str:
    return html.escape(str(s))


def viewer(vid: str, frames: list[str], left: str, right: str,
           mm_per_px_native: float, flagged=(), claude: str = "",
           caption: str = "") -> str:
    """One scrollable series."""

    flag_toggle = ""
    if claude:
        flag_toggle = (f'<label style="text-transform:none;letter-spacing:0">'
                       f'<input type="checkbox" id="{vid}-flag"> show the slices '
                       f'Claude flagged</label>')

    claude_block = ""
    if claude:
        claude_block = (f'<div class="claude"><h4>Claude\'s reading — unverified, '
                        f'not a radiologist</h4>{claude}</div>')

    return f"""
<div class="viewer">
  <div>
    <div class="stage" id="{vid}">
      <img alt="MRI slice" src="{frames[len(frames)//2]}">
      <div class="bar"></div><div class="barlabel"></div>
      <div class="flag">Claude flagged this slice</div>
    </div>
    <div class="axis"><span>&#9664; {_esc(left)}</span><span>{_esc(right)} &#9654;</span></div>
    <div class="note">{caption}</div>
  </div>
  <div class="ctl">
    <label>slice <span class="count" id="{vid}-n"></span></label>
    <input type="range" id="{vid}-sl" min="0" max="{len(frames)-1}" value="{len(frames)//2}">
    <label>brightness</label>
    <input type="range" id="{vid}-b" min="0.4" max="2.2" step="0.05" value="1">
    <label>contrast</label>
    <input type="range" id="{vid}-c" min="0.5" max="2.6" step="0.05" value="1">
    <div class="note">Scroll the image, drag the slider, or click it and use
      <kbd>&larr;</kbd> <kbd>&rarr;</kbd>.</div>
    {flag_toggle}
  </div>
</div>
{claude_block}
<script>mount({json.dumps(vid)}, {json.dumps(frames)}, {json.dumps(list(flagged))},
              {mm_per_px_native});</script>
"""


def resolution_panel_html(panels: list[dict], jpegs: list[str]) -> str:
    cells = []
    for panel, uri in zip(panels, jpegs):
        cells.append(f"""
  <div>
    <h4>{_esc(panel['title'])}</h4>
    <div class="sub">{_esc(panel['sub'])}</div>
    <div class="stage" style="margin-top:8px"><img src="{uri}" alt=""></div>
    <div class="why">{_esc(panel['note'])}</div>
  </div>""")
    return f'<div class="panel3">{"".join(cells)}</div>'


def page(title: str, body: str) -> str:
    """The viewers call `mount` from inline scripts spread through the body, so the
    definition goes in the head. Put it at the end of the body instead and every call
    runs before the function exists — which shows as a page of blank black rectangles,
    with the error only visible in the console."""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(title)}</title><style>{CSS}</style>
<script>{JS}</script></head>
<body><div class="wrap">{body}</div></body></html>"""
