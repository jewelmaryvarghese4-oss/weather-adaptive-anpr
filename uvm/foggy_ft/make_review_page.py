"""
Builds a label-check page for the foggy plate crops: datasets/foggy_ft/review.html (open it in a browser).
Each crop is shown large with my draft reading in an editable box. Mark it right, fix the text, or mark it unreadable / not a plate;
progress is kept in the browser, and "Download" saves foggy_labels_verified.csv (put it in datasets/foggy_ft/).

    python foggy_ft/make_review_page.py
"""
import csv
import json
import os
import shutil

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
SRC = os.path.join(D, "foggy_eval", "full_crop_pipeline")
OUT = os.path.join(D, "foggy_ft")
os.makedirs(os.path.join(OUT, "crops"), exist_ok=True)

rows = list(csv.DictReader(open(os.path.join(SRC, "readings_with_draft_truth.csv"), encoding="utf-8")))
data = []
for r in rows:
    name = "%03d.png" % int(r["number"])
    shutil.copy("%s/crops/%s_plate%s.png" % (SRC, os.path.splitext(r["photo"])[0], r["plate"]), os.path.join(OUT, "crops", name))
    data.append(dict(n=int(r["number"]), img="crops/" + name, photo=r["photo"], plate=int(r["plate"]),
                     draft=r["draft_true_plate"], conf=r["draft_confidence"], raw=r["raw"], clahe=r["clahe"], uvm=r["uvmnet"]))

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Check the foggy plate labels</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--line:#d9dde3;--text:#1c2128;--muted:#6b7380;--ok:#1a7f37;--bad:#c62828;--warn:#b35900;--acc:#1f6feb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1216;--card:#171b21;--line:#2a3038;--text:#e6e9ee;--muted:#8b94a3;--ok:#3fb950;--bad:#ff6b6b;--warn:#f0a04b;--acc:#4c8dff}}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.4 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;z-index:5;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
header b{font-size:16px} .sp{flex:1} button,select{font:inherit}
.btn{border:1px solid var(--line);background:var(--card);color:var(--text);padding:6px 12px;border-radius:7px;cursor:pointer}
.btn.pri{background:var(--acc);border-color:var(--acc);color:#fff}
main{max-width:980px;margin:0 auto;padding:12px 16px 80px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:10px 0;display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:14px}
.card.done{border-left:5px solid var(--ok)} .card.skip{border-left:5px solid var(--warn)} .card.nop{border-left:5px solid var(--bad)}
@media(max-width:760px){.card{grid-template-columns:1fr}}
.imgbox{background:#8a8f98;border-radius:6px;padding:8px;display:flex;align-items:center;justify-content:center;min-height:80px}
.imgbox img{max-width:100%;image-rendering:auto;height:auto;min-width:60%}
.n{font-weight:700;color:var(--acc)} .meta{color:var(--muted);font-size:12.5px}
input[type=text]{width:100%;box-sizing:border-box;font:600 22px/1.2 ui-monospace,Consolas,monospace;letter-spacing:1px;text-transform:uppercase;padding:7px 9px;border:1px solid var(--line);border-radius:7px;background:var(--bg);color:var(--text)}
.row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.tag{font-size:12px;padding:1px 7px;border-radius:9px;border:1px solid var(--line);color:var(--muted)}
.hint{font-size:12.5px;color:var(--muted);margin-top:6px}
</style></head><body>
<header><b>Check the foggy plate labels</b><span id="prog"></span><span class="sp"></span>
<label>Show <select id="flt"><option value="all">all</option><option value="todo">not checked yet</option><option value="sure">my "sure" ones</option><option value="likely">my "likely" ones</option><option value="blank">blank (I could not read)</option></select></label>
<button class="btn pri" id="dl">Download foggy_labels_verified.csv</button></header>
<main><p class="meta">For each crop: if the box shows the plate exactly as you see it, press <b>Right</b>. If not, type the correct plate (letters and digits, no spaces) and press <b>Save</b>. If it is too blurry to read press <b>Unreadable</b>; if it is not a plate press <b>Not a plate</b>. Your answers are kept in this browser; download the file when you are done (you can come back and download again). Raw / CLAHE / UVM are what the models read, shown only as hints.</p><div id="list"></div></main>
<script>
const DATA=__DATA__;
const KEY="foggy_label_review_v1";
let st={}; try{st=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){}};
const $=s=>document.querySelector(s);
function status(d){return (st[d.n]||{}).s||""}
function render(){
  const f=$("#flt").value, L=$("#list"); L.innerHTML="";
  let done=0;
  DATA.forEach(d=>{ if(status(d)) done++; });
  $("#prog").textContent=done+" of "+DATA.length+" checked";
  DATA.forEach(d=>{
    const s=status(d);
    if(f=="todo"&&s) return; if(f=="sure"&&d.conf!="sure") return; if(f=="likely"&&d.conf!="likely") return; if(f=="blank"&&(d.draft||d.conf)) return;
    const cur=(st[d.n]||{}).t!==undefined?st[d.n].t:d.draft;
    const c=document.createElement("div"); c.className="card "+(s=="ok"||s=="fix"?"done":s=="unreadable"?"skip":s=="notplate"?"nop":"");
    c.innerHTML=`<div><div class="meta"><span class="n">#${d.n}</span> &nbsp; photo ${d.photo.slice(9,15)} &nbsp; <span class="tag">my label: ${d.conf||"blank"}</span></div>
      <div class="imgbox"><img src="${d.img}" style="width:100%"></div></div>
      <div><input type="text" id="t${d.n}" value="${cur}" maxlength="12" autocomplete="off" spellcheck="false" placeholder="type the plate">
      <div class="row"><button class="btn pri" data-a="right">Right</button><button class="btn" data-a="save">Save</button><button class="btn" data-a="unreadable">Unreadable</button><button class="btn" data-a="notplate">Not a plate</button></div>
      <div class="hint">raw ${d.raw||"-"} · CLAHE ${d.clahe||"-"} · UVM ${d.uvm||"-"}</div>
      <div class="hint">${s?("saved as: <b>"+s+"</b>"+((st[d.n]||{}).t?(" — "+st[d.n].t):"")):"not checked yet"}</div></div>`;
    c.querySelectorAll("button").forEach(b=>b.onclick=()=>{
      const t=$("#t"+d.n).value.replace(/[^A-Za-z0-9]/g,"").toUpperCase(), a=b.dataset.a;
      if(a=="right"){ if(!t){alert("There is no text in the box. Type the plate, or press Unreadable.");return} st[d.n]={s:"ok",t}; }
      else if(a=="save"){ if(!t){alert("Type the plate first.");return} st[d.n]={s:t==d.draft?"ok":"fix",t}; }
      else if(a=="unreadable") st[d.n]={s:"unreadable",t:""}; else st[d.n]={s:"notplate",t:""};
      save(); render();
    });
    L.appendChild(c);
  });
}
$("#flt").onchange=render;
$("#dl").onclick=()=>{
  const q=v=>'"'+String(v).replace(/"/g,'""')+'"';
  let out="number,photo,plate,label,status\\n";
  DATA.forEach(d=>{const x=st[d.n]; if(x) out+=[d.n,q(d.photo),d.plate,q(x.t||""),x.s].join(",")+"\\n";});
  const a=document.createElement("a"); a.href=URL.createObjectURL(new Blob([out],{type:"text/csv"})); a.download="foggy_labels_verified.csv"; a.click();
};
render();
</script></body></html>"""
open(os.path.join(OUT, "review.html"), "w", encoding="utf-8").write(HTML.replace("__DATA__", json.dumps(data)))
print("wrote", os.path.join(OUT, "review.html"), "with", len(data), "crops")
