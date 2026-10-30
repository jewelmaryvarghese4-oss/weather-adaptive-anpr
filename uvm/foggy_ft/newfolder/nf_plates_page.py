"""results/plates/plates.json + crops -> results/gallery/plates.html (what every model's plate crop reads as, both readers).  python nf_plates_page.py  (any python)"""
import collections
import html
import json
import os

R = "C:/projects/jewwww/datasets/newfoggyphotos/results" if os.name == "nt" else "/mnt/c/projects/jewwww/datasets/newfoggyphotos/results"
J = json.load(open(R + "/plates/plates.json"))
lab = {s["id"]: s["label"] for s in J["sources"]}
rows, summary = [], collections.defaultdict(lambda: [0, 0, 0, 0])      # model -> [plates compared, installed==consensus, weather==consensus]
for P in J["photos"]:
    for pi, pl in enumerate(P["plates"]):
        if len(pl.get("found_on", [])) < 3:       # a box that only one output produced is a false detection
            continue
        reads = pl["reads"]
        cw = collections.Counter(r["weather"] for r in reads.values()).most_common(1)[0][0] if reads else ""
        ci = collections.Counter(r["installed"] for r in reads.values()).most_common(1)[0][0] if reads else ""
        cards = []
        for sid in lab:
            if sid not in reads:
                continue
            r = reads[sid]
            summary[sid][0] += 1
            summary[sid][3] += sid in pl.get("found_on", [])
            summary[sid][1] += r["installed"] == ci
            summary[sid][2] += r["weather"] == cw
            ok = r["weather"] == cw
            cards.append('<figure class="c%s"><a href="../plates/crops/%s_%d_%s.png" target="_blank"><img loading="lazy" src="../plates/crops/%s_%d_%s.png" alt=""></a>'
                         '<figcaption><b>%s</b><br>installed reader: <code>%s</code><br>weather reader: <code>%s</code></figcaption></figure>'
                         % (" ok" if ok else "", P["photo"], pi, sid, P["photo"], pi, sid, html.escape(lab[sid]), html.escape(r["installed"]), html.escape(r["weather"])))
        rows.append('<section><h2>Photo %s — plate %d <small>most common reading: installed <code>%s</code>, weather <code>%s</code></small></h2><div class="g">%s</div></section>'
                    % (P["photo"], pi + 1, html.escape(ci), html.escape(cw), "".join(cards)))
srows = "".join("<tr><td>%s</td><td>%d of %d</td><td>%d</td><td>%d</td></tr>" % (html.escape(lab[k]), v[3], v[0], v[1], v[2]) for k, v in sorted(summary.items(), key=lambda kv: -kv[1][2]))
page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Plate readings per model</title>
<style>:root{--bg:#111418;--p:#1a1f26;--l:#2f3845;--t:#e8edf3;--m:#9aa7b6;--g:#45d483}@media(prefers-color-scheme:light){:root{--bg:#f2f4f7;--p:#fff;--l:#cfd6df;--t:#17202b;--m:#5a6877;--g:#138a4b}}
body{margin:0;padding:14px 16px 60px;background:var(--bg);color:var(--t);font:15px/1.4 system-ui,sans-serif}h1{font-size:18px}h2{font-size:16px;border-bottom:1px solid var(--l);padding-bottom:6px}h2 small{color:var(--m);font-weight:400;font-size:13px;margin-left:8px}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}figure{margin:0;background:var(--p);border:1px solid var(--l);border-radius:8px;overflow:hidden}figure.ok{border-color:var(--g)}
figure img{width:100%;height:96px;object-fit:contain;background:#000;display:block}figcaption{padding:6px 8px;font-size:12.5px;color:var(--m)}figcaption b{color:var(--t)}code{color:var(--t);font-size:13px}
table{border-collapse:collapse;font-size:13px;margin:8px 0 18px}td,th{border:1px solid var(--l);padding:4px 10px;text-align:left}.n{color:var(--m);font-size:13px;max-width:900px}</style></head><body>
<h1>What every model's plate crop reads as</h1>
<p class="n">These photos have no ground truth. Each plate was found on the photo by the plate detector, the same box was cut from every model's output and read by the installed reader and by the weather-trained reader (smoothed or plain view, whichever is more plausible). A green border means the weather reader's reading equals the most common weather-reader reading across all models for that plate — agreement, not proof of being right. Click a crop to open it.</p>
<table><tr><th>Model</th><th>plate found on this output by the detector (conf 0.1)</th><th>installed reader = most common</th><th>weather reader = most common</th></tr>""" + srows + "</table>" + "".join(rows) + "</body></html>"
os.makedirs(R + "/gallery", exist_ok=True)
open(R + "/gallery/plates.html", "w", encoding="utf-8").write(page)
print("plates.html written:", len(rows), "plates")
