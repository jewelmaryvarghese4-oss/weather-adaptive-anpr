"""Appends weeks 17-20 to the project diary (same row format as the earlier weeks) and saves a new copy."""
import copy
from docx import Document

SRC = r"C:\projects\jewwww\documents\Project_Diary_Action_Plan_updated.docx"
OUT = r"C:\projects\jewwww\documents\Project_Diary_Action_Plan_updated2.docx"

WEEKS = [
    (17, "Project report and review presentation (9\u201316 Oct)", [
        "Wrote the intermediate project report in LaTeX for Overleaf: abstract, introduction, the five objectives, the design steps of the system, the conclusion with next steps, and the reference list.",
        "Added the plate-detection section (YOLOv11n trained for 100 epochs on 15,000+ plate images: precision 98.5 %, recall 99.0 %, mAP@50 99.2 %, mAP@50\u201395 85.6 %) and a plate-recognition section with OCR as a sub-subsection covering EasyOCR, PaddleOCR and fastplateOCR, with the result tables in LaTeX.",
        "Built the review presentation (objectives, specification, dehazing methods, CLAHE, LPRNet, real-data results, real-time output, inference), changed its theme to the teal design and checked every number against the experiment logs.",
        "Compiled the ANPR results chapter (23 figures) and the dehazing results report (13 figures) from the test outputs.",
    ]),
    (18, "Hazy Rajasthan test, reader retraining, project clean-up, browser interface and first Ponmudi fog test (17\u201318 Oct)", [
        "Ran UVMNet on a 250-photo hazy Rajasthan subset and YOLOv11 + LPRNet on the restored copy, and checked every reading by eye against the clear photographs: 205 of 248 plates were right on the restored images against 195 of 248 on the hazy originals.",
        "Traced the errors to three gaps: no layout for Delhi plates with a lettered RTO code (DL3CCU5010), the letter Q making up only 0.37 % of real training characters, and red temporary plates never seen. Added the layout and synthetic red-background and Q plates and retrained with 183 verified Rajasthan plates over three seeds: exact reads on 62 held-out crops rose from 48 to 56, 54 and 53, and the best seed was installed after the video and Kerala checks.",
        "Reorganised the project folder (datasets, checkpoints, logs, code backups, documents, archive) with an undo log, and wrote the browser interface (plate_app.py) to pick videos or photos from folders, tested end to end on a video by path, 4K input, photographs, stop and error cases.",
        "Added parallel worker processes to the real-time pipeline (1080p: 40.9 frames per second with one process and 61.8 with four; 4K limited to about 27 by decoding) and made the saved annotated video play at normal speed like the live run.",
        "Ran the installed UVMNet on 60 foggy photographs taken at Ponmudi: the output was dark with a colour cast, so fine-tuning on Ponmudi-style fog was started. Compared raw, CLAHE and UVMNet versions with YOLOv11 + LPRNet, and surveyed the dehazing literature (awesome-dehazing list) against our situation.",
    ]),
    (19, "Real-fog training and comparison of dehazing models at Ponmudi (19\u201320 Oct)", [
        "Labelled the 85 legible plates in the 60 Ponmudi photographs by eye as the test set (test noise about \u00b13 plates), downloaded the GoPro hazy and clear driving videos and the extra Drive folders, and extracted frames to build real-fog training sets, since clear photographs of the same Ponmudi scenes could not be recaptured.",
        "Trained and compared UVMNet variants: a Ponmudi fine-tune with calibrated synthetic fog, a filter-trained model (trained through a guided filter), a plate-aware model (a frozen LPRNet loss through region-of-interest pooling, raising held-out hazy plates from 34 to 39\u201340 of 53), a GoPro-trained model, and a tint and tile-seam fix for the brown cast.",
        "Tested flip, four-way flip and multi-scale averaging, which gave no gain and cost two to five times the time, and showed that the guided filter must be applied after UVMNet because raw network output reads worse.",
        "Built an HTML gallery with every original Ponmudi photograph beside the output of every model, and inspected the outputs visually after each training run.",
        "Trained a fresh UVMNet from random weights on real fog only, with statistics-based losses (tone, dark channel, edges, hue, saturation, grain), exponential-moving-average weights, batch-norm repair, and a preview image and PSNR after every epoch; added an aligned pass with NH-HAZE pairs and synthetic Ponmudi fog.",
        "Result: every dehazer raised the installed reader from 40 to 44\u201349 of 85 plates, but none passed the 47\u201351 of a weather-trained reader on the untouched photographs.",
    ]),
    (20, "Physics-guided dehaze, depth-aware fog, frame fusion, foggy video and report (20\u201321 Oct)", [
        "Added heavy synthetic fog on clear photographs as aligned pairs with ground truth, ran a longer second training, and blended the new model with the GoPro hybrid as a separate variant.",
        "Built the physics-guided dehaze: the network only measures the fog thickness and the veil, and the fog is removed from the real photograph by inverting the scattering law (strong setting, gamma 1.25, with exposure matching). It gave the cleanest-looking pictures, and the installed reader read 49 of 85 plates from them.",
        "Trained a separate depth-aware fog model using Depth Anything V2 depth maps: about 4 dB better on its own held-out test, but no gain on the 60 real photographs.",
        "Tested multi-frame fusion on the Ponmudi videos (133 plate tracks, 9 matched to verified plates): the matched tracks were read by four readers (36 readings), and a single frame gave 1 exact, a vote over frames 9 and the fused image 8\u201310.",
        "Ran the 41 foggy photographs and the 250 synthetic-fog photographs through the best models, ran YOLOv11 + LPRNet on the results, and made three-panel comparisons; built the foggy-video pipeline (UVMNet, physics-guided dehaze, YOLOv11, LPRNet) with slow motion at each plate, an enlarged plate crop and a fixed most-voted reading per vehicle; built a gallery of 43 variants for the 41 photographs.",
        "Wrote the project report in LaTeX (Overleaf-ready, with an editable zip) and built the Phase 1 review presentation on the existing template.",
    ]),
]

d = Document(SRC)
t = d.tables[0]
tmpl_tr = t.rows[-1]._tr
for num, stage, bullets in WEEKS:
    tr = copy.deepcopy(tmpl_tr)
    t._tbl.append(tr)
    row = t.rows[-1]
    # week number and stage
    for cell, val in ((row.cells[0], str(num)), (row.cells[1], stage)):
        p = cell.paragraphs[0]
        p.runs[0].text = val
        for r in p.runs[1:]:
            r.text = ""
    # actual work: one bullet paragraph per item
    c = row.cells[2]
    first = c.paragraphs[0]
    for p in c.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    protos = []
    for i, b in enumerate(bullets):
        if i == 0:
            p = first
        else:
            newp = copy.deepcopy(first._p)
            c._tc.append(newp)
            p = c.paragraphs[-1]
        p.runs[0].text = b
        for r in p.runs[1:]:
            r.text = ""
    # remarks column stays empty
    rc = row.cells[3]
    for p in rc.paragraphs:
        for r in p.runs:
            r.text = ""
d.save(OUT)
print("saved", OUT, len(t.rows), "rows")
