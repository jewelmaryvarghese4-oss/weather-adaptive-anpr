"""Builds the Phase 1 review deck on the user's template (theme, layouts, card styles) with python-pptx."""
import copy, os
from lxml import etree
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_THEME_COLOR as TC
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "img")
TEMPLATE = r"C:\Users\Jewel\Downloads\Weather Adaptive - updated.pptx"
OUT = os.path.join(HERE, "Weather_Adaptive_ANPR_Phase1_Review.pptx")
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main", "p": "http://schemas.openxmlformats.org/presentationml/2006/main"}

TX, BG, A1, A2, A3, A5 = TC.TEXT_1, TC.BACKGROUND_1, TC.ACCENT_1, TC.ACCENT_2, TC.ACCENT_3, TC.ACCENT_5
FOOTER = "Weather Adaptive ANPR \u00b7 Phase 1 review"

prs = Presentation(TEMPLATE)
LIGHT = [l for l in prs.slide_layouts if l.name == "LIGHT"][0]

# ---------------------------------------------------------------- helpers
def set_color(font, color):
    if isinstance(color, RGBColor):
        font.color.rgb = color
    else:
        font.color.theme_color = color


def shadow(shape):
    spPr = shape._element.spPr
    for e in spPr.findall("a:effectLst", NS):
        spPr.remove(e)
    eff = etree.SubElement(spPr, "{%s}effectLst" % NS["a"])
    sh = etree.SubElement(eff, "{%s}outerShdw" % NS["a"], blurRad="76200", dist="19050", dir="5400000", algn="t", rotWithShape="0")
    clr = etree.SubElement(sh, "{%s}srgbClr" % NS["a"], val="0B2B2E")
    etree.SubElement(clr, "{%s}alpha" % NS["a"], val="10000")


def rect(slide, x, y, w, h, fill=BG, adj=0.04, name="Card", line=None, shadowed=True, dash=False):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    s.name = name
    s.adjustments[0] = adj
    if fill is None:
        s.fill.background()
    elif isinstance(fill, RGBColor):
        s.fill.solid(); s.fill.fore_color.rgb = fill
    else:
        s.fill.solid(); s.fill.fore_color.theme_color = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.width = Pt(1.25)
        if isinstance(line, RGBColor):
            s.line.color.rgb = line
        else:
            s.line.color.theme_color = line
        if dash:
            from pptx.enum.dml import MSO_LINE
            s.line.dash_style = MSO_LINE.DASH
    if shadowed:
        shadow(s)
    return s


def R(text, size=16, bold=False, color=TX, font="b", italic=False):
    return dict(text=text, size=size, bold=bold, color=color, font=font, italic=italic)


def P(runs, align="l", after=0, before=0, line=None):
    if isinstance(runs, dict):
        runs = [runs]
    return dict(runs=runs, align=align, after=after, before=before, line=line)


def text(slide, x, y, w, h, paras, anchor="t", name="Text"):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tb.name = name
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = dict(t=MSO_ANCHOR.TOP, m=MSO_ANCHOR.MIDDLE, b=MSO_ANCHOR.BOTTOM)[anchor]
    if isinstance(paras, dict):
        paras = [paras]
    for i, pd in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = dict(l=PP_ALIGN.LEFT, c=PP_ALIGN.CENTER, r=PP_ALIGN.RIGHT)[pd["align"]]
        if pd["after"]:
            p.space_after = Pt(pd["after"])
        if pd["before"]:
            p.space_before = Pt(pd["before"])
        if pd["line"]:
            p.line_spacing = pd["line"]
        for rd in pd["runs"]:
            r = p.add_run()
            r.text = rd["text"]
            f = r.font
            f.size = Pt(rd["size"]); f.bold = rd["bold"]; f.italic = rd["italic"]
            f.name = "+mj-lt" if rd["font"] == "h" else "+mn-lt"
            set_color(f, rd["color"])
    return tb


def badge(slide, x, y, n, color=A1, size=0.5):
    s = rect(slide, x, y, size, size, fill=color, adj=0.16, name="Number badge", shadowed=False)
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.add_run(); r.text = str(n)
    r.font.size = Pt(18); r.font.bold = True; r.font.name = "+mj-lt"; set_color(r.font, BG)
    return s


def stat(slide, x, y, w, h, value, label, color=A1, vsize=26, lsize=14):
    rect(slide, x, y, w, h, adj=0.067, name="Stat card")
    text(slide, x + 0.2, y + 0.12, w - 0.4, 0.55, P(R(value, vsize, True, color, "h")), anchor="m", name="Stat value")
    text(slide, x + 0.2, y + 0.72, w - 0.4, h - 0.8, P(R(label, lsize, False, A5)), name="Stat label")


def picture(slide, fname, x, y, w, h, descr="", pad=0.0):
    path = os.path.join(IMG, fname)
    iw, ih = Image.open(path).size
    s = min(w / iw, h / ih)
    pw, ph = iw * s, ih * s
    pic = slide.shapes.add_picture(path, Inches(x + (w - pw) / 2), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))
    pic.name = "Picture " + fname.split(".")[0]
    pic._element.nvPicPr.cNvPr.set("descr", descr)
    return pic


def add_slide(kicker, title, notes=None):
    s = prs.slides.add_slide(LIGHT)
    for ph in s.placeholders:
        idx = ph.placeholder_format.idx
        if idx == 101:
            ph.text_frame.text = kicker
        elif idx == 102:
            ph.text_frame.text = title
    # slide number placeholder (the layout has one, python-pptx does not copy it)
    src = [sh for sh in LIGHT.placeholders if sh.placeholder_format.type == 13][0]._element
    el = copy.deepcopy(src)
    s.shapes._spTree.append(el)
    if notes:
        ns = s.notes_slide
        if ns.notes_text_frame is None:
            donor = prs.slides[0].notes_slide
            for sh in donor.shapes:
                if sh.is_placeholder and sh.placeholder_format.type in (101, 2):
                    ns.shapes._spTree.append(copy.deepcopy(sh._element))
        ns.notes_text_frame.text = notes
    return s


def table(slide, x, y, w, col_w, rows, header=True, size=14, row_h=0.42, name="Table"):
    nrows, ncols = len(rows), len(rows[0])
    gf = slide.shapes.add_table(nrows, ncols, Inches(x), Inches(y), Inches(w), Inches(row_h * nrows))
    gf.name = name
    tbl = gf.table
    tblPr = tbl._tbl.tblPr
    for a in ("firstRow", "bandRow"):
        if a in tblPr.attrib:
            del tblPr.attrib[a]
    sid = tblPr.find("a:tableStyleId", NS)
    if sid is not None:
        tblPr.remove(sid)
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.14); c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True
            p = tf.paragraphs[0]
            hdr = header and i == 0
            r = p.add_run(); r.text = val
            r.font.size = Pt(size); r.font.bold = hdr or j == 0
            r.font.name = "+mj-lt" if hdr else "+mn-lt"
            set_color(r.font, TX)
            tcPr = c._tc.get_or_add_tcPr()
            for tag in ("lnL", "lnR", "lnT", "lnB"):
                ln = etree.SubElement(tcPr, "{%s}%s" % (NS["a"], tag), w="9525", cap="flat", cmpd="sng", algn="ctr")
                sf = etree.SubElement(ln, "{%s}solidFill" % NS["a"])
                etree.SubElement(sf, "{%s}srgbClr" % NS["a"], val="CFE0DC")
                etree.SubElement(ln, "{%s}prstDash" % NS["a"], val="solid")
            sf = etree.SubElement(tcPr, "{%s}solidFill" % NS["a"])
            etree.SubElement(sf, "{%s}srgbClr" % NS["a"], val="DDEBE8" if hdr else "FFFFFF")
    return gf


def style_chart(chart, size=12):
    chart.font.size = Pt(size)
    chart.font.name = "Calibri"
    chart.font.color.theme_color = A5


# ---------------------------------------------------------------- footer text on the layouts
for lay in prs.slide_layouts:
    for sh in lay.shapes:
        if sh.name == "Footer" and sh.has_text_frame and sh.text_frame.paragraphs[0].runs:
            runs = sh.text_frame.paragraphs[0].runs
            runs[0].text = FOOTER
            for r in runs[1:]:
                r.text = ""

# ---------------------------------------------------------------- keep slides 1-3, drop 4-9
sldIdLst = prs.slides._sldIdLst
ids = list(sldIdLst)
for sid in ids[3:]:
    prs.part.drop_rel(sid.rId)
    sldIdLst.remove(sid)

# cover: change the kicker
cover = prs.slides[0]
for sh in cover.placeholders:
    if sh.placeholder_format.idx == 101:
        r = sh.text_frame.paragraphs[0].runs
        r[0].text = "PROJECT PRESENTATION \u00b7 PHASE 1 REVIEW"
        for x in r[1:]:
            x.text = ""
cover.notes_slide.notes_text_frame.text = ("Weather Adaptive ANPR System: Phase 1 review covering the problem, the design, the partial outputs, "
                                           "the time plan for Phase 2 and the future directions.")

# slide 2 objectives: unchanged except notes
prs.slides[1].notes_slide.notes_text_frame.text = "Five objectives. Objectives 1 and 2 are the subject of Phase 1; objectives 3 to 5 are Phase 2."

# slide 3: spec table, update two rows
spec = prs.slides[2]
for sh in spec.shapes:
    if sh.has_table:
        t = sh.table
        for row in t.rows:
            key = row.cells[0].text_frame.text.strip()
            def setcell(cell, new):
                p = cell.text_frame.paragraphs[0]
                p.runs[0].text = new
                for x in p.runs[1:]:
                    x.text = ""
            if key == "Weather restoration":
                setcell(row.cells[1], "UVMNet with physics-guided fog removal; lightweight U-Net deraining for rain; CLAHE for low contrast")
            elif key == "Edge hardware":
                setcell(row.cells[1], "Jetson Nano-class module, HDR / STARVIS-class camera, NoIR plate camera with IR illuminator")
spec.notes_slide.notes_text_frame.text = "Design values of the camera node. The weather restoration row now names the models that were built in Phase 1."

# ================================================================ S4 problem
s = add_slide("BACKGROUND \u00b7 THE PROBLEM", "Fog and rain break plate recognition",
              "A general-purpose OCR read only 16.5 to 25.2 percent of plates exactly in fog and 5.9 to 11 percent in rain. "
              "Three things fail together: the image, the plate size and the reader.")
stat(s, 0.6, 1.85, 3.89, 1.45, "16.5\u201325.2 %", "of plates read exactly in fog by a general-purpose OCR (EasyOCR)", A2)
stat(s, 4.72, 1.85, 3.89, 1.45, "5.9\u201311.0 %", "of plates read exactly in rain by the same OCR", A2)
stat(s, 8.84, 1.85, 3.89, 1.45, "22\u201339 px", "width of two-wheeler and auto plates in traffic video", A1)
cards = [
    ("The image is degraded", "Fog lowers contrast and adds a veil of scattered light; rain adds streaks and spray; low light adds noise."),
    ("The plate is small", "A plate 10\u201320 m away may be only 25\u201340 pixels wide; two-line plates on two-wheelers and autos are smaller still."),
    ("The reader is brittle", "Networks trained on clean plates misread blurred characters, drop repeated digits and confuse Q, O and 0."),
]
for i, (h, b) in enumerate(cards):
    x = 0.6 + i * 4.12
    rect(s, x, 3.55, 3.89, 2.4, adj=0.026, name="Problem card")
    badge(s, x + 0.25, 3.78, i + 1, [A1, A2, A3][i])
    text(s, x + 0.95, 3.78, 2.8, 0.5, P(R(h, 18, True, TX, "h")), anchor="m", name="Problem heading")
    text(s, x + 0.25, 4.5, 3.4, 1.4, P(R(b, 16)), name="Problem text")
text(s, 0.6, 6.2, 12.13, 0.45, P([R("The system must therefore ", 15, False, A5), R("restore degraded frames, read plates in real time and handle Indian plate formats.", 15, True, A1)]), anchor="m", name="Takeaway")

# ================================================================ S5 literature
s = add_slide("LITERATURE REVIEW", "What the project builds on",
              "Four strands of work: detection, recognition, weather restoration and multi-frame evidence. Each has an open problem that Phase 1 addresses.")
lit = [
    ("Plate detection", "YOLO family: boxes in one pass. YOLOv11 (C3k2 blocks); the nano size fits embedded hardware. CCPD and RodoSol-ALPR datasets.",
     "Small plates and degraded images."),
    ("Plate recognition", "EasyOCR is a scene-text reader. LPRNet with CTC loss reads the whole plate in one pass, compactly.",
     "Repeated characters are shortened; Indian plates are scarce in public data."),
    ("Weather restoration", "DCP, CLAHE; learned dehazers (RESIDE, Foggy Cityscapes, NH-HAZE); UVM-Net, a Mamba U-Net; NeRD-Rain and NAFNet for rain.",
     "Synthetic-to-real gap; PSNR gains do not mean better reading."),
    ("Multi-frame evidence", "ECC alignment of crops; tracking and voting over the frames of one vehicle.",
     "Needs a real-time implementation for live video."),
]
for i, (h, how, gap) in enumerate(lit):
    x = 0.6 + i * 3.1
    rect(s, x, 1.85, 2.93, 4.85, adj=0.03, name="Literature card")
    badge(s, x + 0.22, 2.05, i + 1, [A1, A2, A3, A5][i])
    text(s, x + 0.85, 2.05, 2.0, 0.5, P(R(h, 16, True, TX, "h")), anchor="m", name="Literature heading")
    text(s, x + 0.22, 2.8, 2.5, 2.4, [P(R("USED HERE", 13, True, A1), after=3), P(R(how, 15))], name="Literature detail")
    text(s, x + 0.22, 5.3, 2.5, 1.35, [P(R("OPEN PROBLEM", 13, True, A2), after=3), P(R(gap, 15))], name="Literature gap")

# ================================================================ S6 architecture
s = add_slide("SYSTEM DESIGN \u00b7 ARCHITECTURE", "Two layers: a camera node and a central system",
              "The camera node turns video into confirmed plate numbers locally and sends only the plate number, a crop, the camera identifier and the time. "
              "The central system verifies, alerts, dispatches and correlates. Phase 1 covers the camera node.")
# camera node container
rect(s, 0.6, 1.85, 12.13, 2.0, fill=BG, adj=0.04, name="Node container", line=A1)
text(s, 0.85, 1.93, 6, 0.35, P([R("CAMERA NODE", 14, True, A1, "h"), R("   Phase 1 \u00b7 implemented and tested", 14, False, A5)]), anchor="m", name="Node label")
node = [("Camera", "capture"), ("Plate detection", "YOLOv11n"), ("Weather restoration", "UVMNet \u00b7 deraining"), ("Plate reading", "LPRNet + format rules"), ("Tracking", "and voting")]
bw, gap = 2.1, 0.29
for i, (a, b) in enumerate(node):
    x = 0.85 + i * (bw + gap)
    rect(s, x, 2.4, bw, 1.2, fill=A1, adj=0.08, name="Node block", shadowed=False)
    text(s, x + 0.08, 2.45, bw - 0.16, 1.1, [P(R(a, 15, True, BG, "h"), align="c"), P(R(b, 13, False, BG), align="c")], anchor="m", name="Node block text")
    if i < len(node) - 1:
        ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + bw + 0.04), Inches(2.85), Inches(gap - 0.08), Inches(0.3))
        ar.name = "Arrow"; ar.fill.solid(); ar.fill.fore_color.theme_color = A5; ar.line.fill.background()
# transfer label
text(s, 0.6, 3.95, 12.13, 0.35, P([R("\u25bc  plate number \u00b7 plate crop \u00b7 camera ID \u00b7 time", 14, True, A5)], align="c"), anchor="m", name="Transfer label")
# central container
rect(s, 0.6, 4.35, 12.13, 2.35, fill=BG, adj=0.04, name="Central container", line=A2, dash=True)
text(s, 0.85, 4.43, 7, 0.35, P([R("CENTRAL SYSTEM", 14, True, A2, "h"), R("   Phase 2 \u00b7 planned", 14, False, A5)]), anchor="m", name="Central label")
central = [("Vehicle database", "registration, insurance, PUC, stolen, blacklist"), ("Status classification", "authorised \u00b7 non-compliant \u00b7 unauthorised \u00b7 blacklisted \u00b7 stolen"),
           ("Dashboard and alerts", "details, image, timestamp"), ("Dispatch", "nearest officer by road distance"), ("Multi-node correlation", "convoy, route anomaly, loitering")]
cw = 2.24
for i, (a, b) in enumerate(central):
    x = 0.85 + i * (cw + 0.115)
    rect(s, x, 4.9, cw, 1.6, fill=BG, adj=0.06, name="Central block", line=A2)
    text(s, x + 0.1, 4.97, cw - 0.2, 1.5, [P(R(a, 15, True, A2, "h"), align="c", after=3), P(R(b, 13, False, TX), align="c")], anchor="m", name="Central block text")

# ================================================================ S7 camera node design
s = add_slide("SYSTEM DESIGN \u00b7 CAMERA NODE", "One frame, from pixels to a confirmed plate",
              "Frame flow in the node and three design decisions: the order of restoration and detection, the select view, and the choice of Jetson over Raspberry Pi 5.")
flow = ["Frame", "Weather check", "Restoration: fog \u00b7 rain \u00b7 low contrast", "YOLOv11n plate boxes", "Crop, flatten two-line", "LPRNet + format decode", "Track and vote", "Confirmed plate"]
cw, ch, gx = 2.8, 0.82, 0.31
for i, t in enumerate(flow):
    r_, c_ = divmod(i, 4)
    x = 0.6 + c_ * (cw + gx)
    y = 1.9 + r_ * 1.1
    last = i == len(flow) - 1
    rect(s, x, y, cw, ch, fill=A1 if last else BG, adj=0.12, name="Flow step")
    text(s, x + 0.12, y, cw - 0.24, ch, P(R(t, 14, True, BG if last else TX, "h"), align="c"), anchor="m", name="Flow text")
    if c_ < 3:
        ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x + cw + 0.03), Inches(y + 0.26), Inches(gx - 0.06), Inches(0.3))
        ar.name = "Arrow"; ar.fill.solid(); ar.fill.fore_color.theme_color = A5; ar.line.fill.background()
dec = [
    ("Restore first, or detect first", "Whole frame restored before detection gives the clearest picture (foggy video). Detecting first and restoring only the crop is cheaper (real-time loop)."),
    ("Select view", "The reader chooses between the plain and the smoothed crop, because heavy restoration of a crop can hurt reading."),
    ("Jetson, not Raspberry Pi 5", "The Mamba kernel of UVMNet is CUDA-only. Latency on the Jetson module is a Phase 2 test."),
]
for i, (h, b) in enumerate(dec):
    x = 0.6 + i * 4.12
    rect(s, x, 4.25, 3.89, 2.45, adj=0.03, name="Decision card")
    text(s, x + 0.25, 4.42, 3.4, 0.4, P(R(h, 18, True, [A1, A2, A3][i], "h")), name="Decision heading")
    text(s, x + 0.25, 4.95, 3.4, 1.7, P(R(b, 15)), name="Decision text")

# ================================================================ S8 detector
s = add_slide("PARTIAL OUTPUT \u00b7 PLATE DETECTION", "YOLOv11n finds 99 % of plates",
              "Trained for 100 epochs on more than 15,000 annotated plate images. A fine-tune on traffic frames raised recall on 23 frames from 74 to 87 percent "
              "but was not adopted because the reader got fewer plates right with its boxes.")
sx = [(0.6, 1.85), (3.4, 1.85), (0.6, 3.1), (3.4, 3.1)]
sv = [("98.5 %", "Precision", A1), ("99.0 %", "Recall", A1), ("99.2 %", "mAP@50", A2), ("85.6 %", "mAP@50\u201395", A1)]
for (x, y), (v, l, c) in zip(sx, sv):
    stat(s, x, y, 2.65, 1.1, v, l, c, vsize=26, lsize=14)
rect(s, 0.6, 4.35, 5.45, 2.35, adj=0.04, name="Detail card")
picture(s, "yolo_predictions.jpg", 0.75, 4.5, 2.05, 2.05, "Detector predictions on validation images")
text(s, 3.0, 4.5, 2.9, 2.1, [P(R("Setup", 15, True, A1), after=3),
                              P(R("100 epochs, 15,000+ images; run at 1280 px, confidence 0.25 so small plates in 1080p and 4K are kept.", 14))], name="Detector text")
rect(s, 6.3, 1.85, 6.43, 4.85, adj=0.03, name="Chart card")
text(s, 6.55, 1.97, 6.0, 0.35, P(R("Training and validation curves", 14, True, TX, "h")), name="Image title")
picture(s, "yolo_curves.png", 6.5, 2.45, 6.03, 3.4, "Training curves of the YOLOv11n detector")
text(s, 6.55, 5.95, 5.95, 0.7, P(R("Losses fall steadily and the validation metrics level off; 99 % of plates are found and about 1 % of background regions are wrongly detected.", 12, False, A5)), name="Caption")

# ================================================================ S9 first restoration comparison
s = add_slide("PARTIAL OUTPUT \u00b7 FIRST RESTORATION COMPARISON", "DCP, a transformer network and CLAHE compared",
              "Tested on 1,001 hazy plate photographs and a real foggy road photo; the numbers are the actual EasyOCR readings. "
              "None of the three recovers detail that the camera did not capture.")
cols = [
    ("Dark Channel Prior", "Helped on mild haze", A3, "dcp_mild.png", "Plate reading confidence rose from 0.227 to 0.336 (about 48 %). On dense fog it did nothing, or with strong settings turned the frame white."),
    ("Transformer U-Net", "Clean look, worse reading", A2, "neural_vs_clahe.png", "Best-looking car, but plate edges were smoothed away: mean OCR confidence 0.104 against 0.326 for the hazy input."),
    ("CLAHE", "Recovered a plate", A1, "clahe_recovered.png", "Read \u201cKL 55R 2473\u201d at 0.774 where the input and DCP found nothing; but it hurt another plate (0.227 to 0.097)."),
]
for i, (name_, verdict, c, img, body) in enumerate(cols):
    x = 0.6 + i * 4.12
    rect(s, x, 1.85, 3.89, 3.95, adj=0.026, name="Method card")
    text(s, x + 0.25, 2.0, 3.4, 0.4, P(R(name_, 18, True, TX, "h")), name="Method name")
    text(s, x + 0.25, 2.42, 3.4, 0.3, P(R(verdict, 15, True, c, "h")), name="Verdict")
    picture(s, img, x + 0.2, 2.85, 3.49, 1.45, name_ + " example")
    text(s, x + 0.25, 4.4, 3.4, 1.4, P(R(body, 15)), name="Method text")
rect(s, 0.6, 6.0, 12.13, 0.7, adj=0.12, name="Note card")
text(s, 0.85, 6.0, 11.6, 0.7, P([R("CLAHE was chosen as the first route: ", 15, True, A1), R("fastest (about 1.3 ms against 17 ms for DCP) and never fails badly.", 15)]), anchor="m", name="Note")

# ================================================================ S10 UVMNet
s = add_slide("PARTIAL OUTPUT \u00b7 UVMNET FOG REMOVAL", "UVMNet trained in stages for fog removal",
              "UVMNet is a Mamba-based U-shaped network with 116.2 million parameters, run in 1536-pixel tiles. Each stage answered a problem found in the previous one.")
rect(s, 0.6, 1.85, 5.6, 4.85, adj=0.03, name="Table card")
rows = [["Stage", "Result"],
        ["A. Traffic haze\n8,925 pairs", "22.95 \u2192 32.96 dB; green cast on our own photos"],
        ["B. Plate fine-tune\n250 crops, 1,968 variants", "Brightness kept 46 % \u2192 70 %"],
        ["C. Multi-dataset\nOTS + Foggy-Hazy + NH-HAZE", "27.30 dB; 9 of 10 real photos clean"],
        ["D. Variants tried\nlow-light, CLAHE input", "Unstable or worse; dropped"]]
table(s, 0.75, 2.0, 5.3, [2.55, 2.75], rows, size=13, row_h=0.88, name="Stage table")
rect(s, 6.45, 1.85, 6.28, 4.05, adj=0.03, name="Image card")
picture(s, "syn_pairs.png", 6.55, 1.95, 6.08, 3.55, "Synthetic fog: input left, output right, two plate photographs")
text(s, 6.45, 5.52, 6.28, 0.3, P(R("Synthetic fog on plate photographs: input (left) and output (right)", 12, False, A5), align="c"), name="Caption")
rect(s, 6.45, 6.05, 3.05, 0.65, adj=0.15, name="Chip")
text(s, 6.6, 6.05, 2.8, 0.65, [P([R("116.2 M ", 18, True, A1, "h"), R("parameters", 13, False, A5)])], anchor="m", name="Chip text")
rect(s, 9.68, 6.05, 3.05, 0.65, adj=0.15, name="Chip")
text(s, 9.83, 6.05, 2.8, 0.65, [P([R("205 vs 195 ", 18, True, A2, "h"), R("of 248 right", 13, False, A5)])], anchor="m", name="Chip text")

# ================================================================ S11 Ponmudi
s = add_slide("PARTIAL OUTPUT \u00b7 REAL FOG AT PONMUDI", "Physics-guided dehaze on real fog photos",
              "60 photographs at Ponmudi, 85 verified plates. The network only measures the fog; the scattering law is inverted on the real photo. "
              "Pictures look clean, but plate reading stays within the noise of the test.")
rect(s, 0.6, 1.85, 6.95, 4.85, adj=0.03, name="Image card")
picture(s, "pon_1.jpg", 0.7, 1.95, 6.75, 2.34, "Original, GoPro hybrid with tint fix, physics-guided: photo 1")
picture(s, "pon_20.jpg", 0.7, 4.35, 6.75, 2.34, "Original, GoPro hybrid with tint fix, physics-guided: photo 20")
steps = [("Measure", "UVMNet output gives the fog thickness t (least squares, guided filter)"),
         ("Veil", "Atmospheric light A from the brightest 0.1 % of the dark channel"),
         ("Invert", "J = (I \u2212 A) / t^\u03b3 + A, with a floor on t and a cap on detail gain"),
         ("Match", "Exposure match so removing the veil does not darken the picture")]
for i, (h, b) in enumerate(steps):
    y = 1.85 + i * 0.68
    rect(s, 7.8, y, 4.93, 0.6, adj=0.12, name="Step row")
    badge(s, 7.9, y + 0.07, i + 1, [A1, A2, A3, A5][i], size=0.46)
    text(s, 8.5, y, 4.15, 0.6, [P([R(h + "  ", 14, True, TX, "h"), R(b, 12, False, A5)])], anchor="m", name="Step text")
stat(s, 7.8, 4.7, 2.4, 1.3, "40 \u2192 49", "of 85 plates, installed reader, untouched vs physics-guided", A1, vsize=24, lsize=12)
stat(s, 10.33, 4.7, 2.4, 1.3, "47\u201351", "of 85, weather-trained reader on untouched photos", A2, vsize=24, lsize=12)
text(s, 7.8, 6.12, 4.93, 0.6, P(R("The limit is the information in the photo, not the dehazer (test noise \u00b13 plates).", 13, True, A1)), name="Takeaway")

# ================================================================ S12 deraining
s = add_slide("PARTIAL OUTPUT \u00b7 DERAINING", "A lightweight network for rain",
              "A5 is a U-Net-style network with about 1.72 million parameters trained from scratch on RodoSol-ALPR images with synthetic rain. "
              "On real rain some streaks are reduced but dense rain, glare and spray remain; the effect on plate reading is still to be shown.")
rect(s, 0.6, 1.85, 5.9, 4.85, adj=0.03, name="Chart card")
text(s, 0.85, 1.97, 5.4, 0.35, P(R("PSNR on the 30-image synthetic rain test (dB)", 14, True, TX, "h")), name="Chart title")
cd = CategoryChartData()
cd.categories = ["No deraining", "NeRD-Rain", "A4", "A5"]
cd.add_series("PSNR (dB)", (22.17, 20.98, 25.61, 29.19))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.75), Inches(2.4), Inches(5.6), Inches(4.2), cd)
gf.name = "Deraining chart"
ch = gf.chart
style_chart(ch, 12)
ch.has_legend = False; ch.has_title = False
plot = ch.plots[0]; plot.gap_width = 60; plot.has_data_labels = True
plot.data_labels.show_value = True; plot.data_labels.number_format = "0.00"; plot.data_labels.number_format_is_linked = False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size = Pt(13); plot.data_labels.font.bold = True
ser = plot.series[0]
for i in range(4):
    pt = ser.points[i]; pt.format.fill.solid()
    if i == 3:
        pt.format.fill.fore_color.theme_color = A1
    else:
        pt.format.fill.fore_color.rgb = RGBColor(0xA9, 0xB9, 0xB5)
ch.value_axis.minimum_scale = 16; ch.value_axis.maximum_scale = 32; ch.value_axis.major_unit = 4
ch.value_axis.has_major_gridlines = True
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xEB, 0xE8)
ch.value_axis.format.line.fill.background()
ch.value_axis.tick_labels.font.size = Pt(12); ch.category_axis.tick_labels.font.size = Pt(12)
rect(s, 6.75, 1.85, 5.98, 2.3, adj=0.03, name="Image card")
picture(s, "derain_compare_nerd_a5.jpg", 6.85, 1.95, 5.78, 2.1, "Same rainy image: original, NeRD-Rain, A5")
text(s, 6.75, 4.2, 5.98, 0.3, P(R("Original \u00b7 NeRD-Rain \u00b7 A5 on the same rainy photo", 12, False, A5), align="c"), name="Caption")
stat(s, 6.75, 4.65, 2.9, 1.2, "+7.02 dB", "A5 over the unprocessed input", A1, vsize=24, lsize=12)
stat(s, 9.83, 4.65, 2.9, 1.2, "1.72 M", "parameters, small enough for the edge", A2, vsize=24, lsize=12)
text(s, 6.75, 6.0, 5.98, 0.7, P(R("PW-FNet reached 36.46 dB on synthetic plates but gave almost no gain on real rain (23.64 vs 23.60 dB). Real-rain benefit is a Phase 2 item.", 12, False, A5)), name="Note")

# ================================================================ S13 LPRNet
s = add_slide("PARTIAL OUTPUT \u00b7 PLATE READER", "LPRNet improved step by step with real data",
              "Character error fell from 0.70 to 0.249 with synthetic pre-training and real plates. Each later step added a kind of data the reader lacked.")
rect(s, 0.6, 1.85, 6.0, 4.85, adj=0.03, name="Chart card")
text(s, 0.85, 1.97, 5.5, 0.35, P(R("Character error rate (lower is better)", 14, True, TX, "h")), name="Chart title")
cd = CategoryChartData()
cd.categories = ["From scratch", "Synthetic pre-training + real", "+ CCPD and mixing"]
cd.add_series("CER", (0.70, 0.32, 0.249))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.75), Inches(2.4), Inches(5.7), Inches(4.2), cd)
gf.name = "CER chart"
ch = gf.chart; style_chart(ch, 12)
ch.has_legend = False; ch.has_title = False
plot = ch.plots[0]; plot.gap_width = 60; plot.has_data_labels = True
plot.data_labels.show_value = True; plot.data_labels.number_format = "0.000"; plot.data_labels.number_format_is_linked = False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
plot.data_labels.font.size = Pt(13); plot.data_labels.font.bold = True
for i in range(3):
    pt = plot.series[0].points[i]; pt.format.fill.solid()
    if i == 2:
        pt.format.fill.fore_color.theme_color = A1
    else:
        pt.format.fill.fore_color.rgb = RGBColor(0xA9, 0xB9, 0xB5)
ch.value_axis.minimum_scale = 0; ch.value_axis.maximum_scale = 0.8
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xEB, 0xE8)
ch.value_axis.format.line.fill.background()
ch.value_axis.tick_labels.font.size = Pt(12); ch.category_axis.tick_labels.font.size = Pt(12)
st = [("6.7 \u2192 9.0 of 16", "Kerala held-out plates exact, after adding the Zenodo set", A1),
      ("0.50 \u2192 0.31", "character error on small video plates, with degradation training", A2),
      ("38 \u2192 41 of 51", "plates exact on six unseen videos, after real-video crops", A1),
      ("48 \u2192 56 of 62", "held-out crops exact, after Rajasthan, red-plate and Q data", A2)]
for i, (v, l, c) in enumerate(st):
    stat(s, 6.85, 1.85 + i * 1.23, 5.88, 1.1, v, l, c, vsize=24, lsize=13)

# ================================================================ S14 photos
s = add_slide("PARTIAL OUTPUT \u00b7 FULL PIPELINE ON PHOTOS", "YOLOv11 + LPRNet in fog and in good weather",
              "Examples of the full pipeline on photographs. Top row: real fog at Ponmudi after the physics-guided dehaze. Bottom row: good weather. "
              "These are examples where the plate is large enough; small or blurred plates are still missed.")
rect(s, 0.6, 1.85, 6.75, 4.85, adj=0.03, name="Image card")
for k, fn in enumerate(["ph_fog1.jpg", "ph_fog2.jpg", "ph_clr1.jpg", "ph_clr2.jpg"]):
    picture(s, fn, 0.7 + (k % 2) * 3.3, 1.95 + (k // 2) * 2.4, 3.25, 2.35, "Detected plate and reading, " + fn.split(".")[0])
items = [("Fog (top row)", "KL23Z3195 and KA03NT1748 read exactly on Ponmudi photos after the physics-guided dehaze.", A1),
         ("Good weather (bottom row)", "TN07BU5427 and MH14EH7958 read exactly.", A3),
         ("Format-constrained decoding", "Two letters, two digits, one or two letters, four digits. The network\u2019s own probabilities choose the characters each slot allows, so Q survives; signs and logos are dropped.", A2)]
y = 1.85
for h, b, c in items:
    hh = 1.1 if len(b) < 100 else 2.0
    rect(s, 7.6, y, 5.13, hh, adj=0.05, name="Info card")
    text(s, 7.85, y + 0.12, 4.65, 0.35, P(R(h, 16, True, c, "h")), name="Info heading")
    text(s, 7.85, y + 0.5, 4.65, hh - 0.55, P(R(b, 14)), name="Info text")
    y += hh + 0.18
text(s, 7.6, y + 0.02, 5.13, 0.6, P(R("Box labels show the decoded number and the detector confidence.", 12, False, A5)), name="Caption")

# ================================================================ S15 real time + foggy video
s = add_slide("PARTIAL OUTPUT \u00b7 REAL TIME AND FOGGY VIDEO", "Real-time reading and voting over frames",
              "Detection, tracking, reading and voting run live. Parallel worker processes raise throughput until 4K decoding becomes the limit. "
              "On the foggy clips each vehicle gets one fixed, most-voted reading.")
rect(s, 0.6, 1.85, 6.3, 4.85, adj=0.03, name="Chart card")
text(s, 0.85, 1.97, 5.8, 0.35, P(R("Throughput (frames per second) by worker processes", 14, True, TX, "h")), name="Chart title")
cd = CategoryChartData()
cd.categories = ["1 process", "2 workers", "4 workers", "6 workers"]
cd.add_series("1080p, 60 fps", (40.9, 51.5, 61.8, 47.0))
cd.add_series("4K", (13.5, 19.7, 19.0, 16.5))
gf = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.75), Inches(2.4), Inches(6.0), Inches(4.2), cd)
gf.name = "Throughput chart"
ch = gf.chart; style_chart(ch, 12)
ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False; ch.legend.font.size = Pt(12)
ch.has_title = False
plot = ch.plots[0]; plot.gap_width = 70; plot.overlap = -5; plot.has_data_labels = True
plot.data_labels.show_value = True; plot.data_labels.number_format = "0.0"; plot.data_labels.number_format_is_linked = False
plot.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END; plot.data_labels.font.size = Pt(12)
plot.series[0].format.fill.solid(); plot.series[0].format.fill.fore_color.theme_color = A1
plot.series[1].format.fill.solid(); plot.series[1].format.fill.fore_color.rgb = RGBColor(0xA9, 0xB9, 0xB5)
ch.value_axis.minimum_scale = 0; ch.value_axis.maximum_scale = 75
ch.value_axis.major_gridlines.format.line.color.rgb = RGBColor(0xDD, 0xEB, 0xE8)
ch.value_axis.format.line.fill.background()
ch.value_axis.tick_labels.font.size = Pt(12); ch.category_axis.tick_labels.font.size = Pt(12)
stat(s, 7.15, 1.85, 2.7, 1.2, "\u2264 55 fps", "real time at 1080p on the laptop GPU", A1, vsize=24, lsize=12)
stat(s, 10.03, 1.85, 2.7, 1.2, "10 of 12", "plates exact on an unseen 4K traffic video", A2, vsize=24, lsize=12)
rect(s, 7.15, 3.25, 5.58, 3.45, adj=0.035, name="Table card")
text(s, 7.4, 3.35, 5.1, 0.35, P(R("Foggy Ponmudi clip: vehicles with fixed readings", 14, True, TX, "h")), name="Table title")
rows = [["Plate read", "Votes", "Share"],
        ["KL25C2021", "29 of 42", "69 %"],
        ["KL01CH7254", "30 of 48", "62 %"],
        ["KL01BJ5436", "31 of 68", "46 %"],
        ["KL16N8684", "39 of 113", "35 %"]]
table(s, 7.4, 3.8, 5.1, [2.3, 1.5, 1.3], rows, size=13, row_h=0.46, name="Vehicle table")
text(s, 7.4, 6.15, 5.1, 0.5, P(R("A low share (here 35\u201346 %) marks readings to confirm.", 12, False, A5)), name="Table note")

# ================================================================ S16 inference
s = add_slide("INFERENCE", "What the results so far tell us",
              "Six conclusions from Phase 1.")
items = [
    ("The detector is reliable.", " YOLOv11n reaches 99.2 % mAP@50 on 15,000+ plate images."),
    ("A plate-specific reader was needed.", " EasyOCR read 16.5\u201325.2 % of fog plates; LPRNet cut character error from 0.70 to 0.249 and reads 41 of 51 plates on unseen videos."),
    ("Dehazing helps the viewer more than the reader.", " UVMNet and the physics-guided step give clean pictures, but plates read on real fog stay within the test noise."),
    ("Combining frames is the lever.", " On Ponmudi tracks a single frame read 1 of 36 plates and a vote over frames 9; the pipeline runs live at up to 55 fps."),
    ("Known gaps.", " Repeated digits, other states\u2019 series, small or blurred plates; benefit of deraining on real rain is unproven."),
    ("Next:", " vehicle database and alerts, dispatch, multi-node correlation and embedded deployment."),
]
for i, (b, t) in enumerate(items):
    y = 1.85 + i * 0.8
    rect(s, 0.6, y, 12.13, 0.7, adj=0.1, name="Inference row")
    text(s, 0.8, y, 0.5, 0.7, P(R(str(i + 1), 20, True, A2 if i % 2 else A1, "h")), anchor="m", name="Number")
    text(s, 1.45, y, 11.1, 0.7, P([R(b, 15, True, TX), R(t, 15, False, TX)]), anchor="m", name="Inference text")

# ================================================================ S17 time plan
s = add_slide("TIME PLAN", "Phase 1 is done; Phase 2 is planned to April 2027",
              "Proposed schedule for Phase 2, to be confirmed with the guide. Phase 1 milestones are summarised by the weeks of the action plan.")
# phase 1 strip
text(s, 0.6, 1.8, 6, 0.3, P(R("PHASE 1 \u00b7 COMPLETED", 13, True, A1)), name="Phase 1 label")
ph1 = [("Weeks 1\u20134", "Study, literature, set-up"), ("Weeks 5\u201310", "Detector, first dehazing"), ("Weeks 11\u201314", "UVMNet, LPRNet"), ("Weeks 15\u201316", "Real data, real time"), ("Week 17+", "Real fog, deraining, report")]
for i, (a, b) in enumerate(ph1):
    x = 0.6 + i * 2.45
    rect(s, x, 2.15, 2.35, 0.95, fill=A1, adj=0.1, name="Phase 1 block", shadowed=False)
    text(s, x + 0.08, 2.15, 2.2, 0.95, [P(R(a, 13, True, BG, "h"), align="c"), P(R(b, 13, False, BG), align="c")], anchor="m", name="Phase 1 text")
# gantt
text(s, 0.6, 3.25, 8, 0.3, P([R("PHASE 2 \u00b7 PROPOSED SCHEDULE", 13, True, A2), R("   to be confirmed with the guide", 12, False, A5)]), name="Phase 2 label")
gx0, lw, mw = 0.6, 4.3, 1.27
months = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]
rect(s, gx0, 3.55, 12.13, 3.2, adj=0.02, name="Gantt card")
for j, m in enumerate(months):
    text(s, gx0 + lw + j * mw, 3.6, mw, 0.3, P(R(m, 13, True, A5), align="c"), anchor="m", name="Month")
rows = [("Vehicle database and status classification", 0, 2, A1),
        ("Dashboard, alerts and dispatch", 1, 3, A1),
        ("Multi-node correlation", 2, 4, A1),
        ("Embedded port to Jetson and latency tests", 2, 5, A2),
        ("Real-rain and night evaluation; weather router", 0, 3, A3),
        ("More labelled footage; reader improvements", 0, 5, A3),
        ("Field tests, final evaluation, report and demo", 4, 6, A5)]
for i, (lab, a, b, c) in enumerate(rows):
    y = 3.95 + i * 0.38
    text(s, gx0 + 0.2, y, lw - 0.3, 0.32, P(R(lab, 13)), anchor="m", name="Task")
    bar = rect(s, gx0 + lw + a * mw + 0.04, y + 0.04, (b - a) * mw - 0.08, 0.24, fill=c, adj=0.3, name="Task bar", shadowed=False)

# ================================================================ S18 future
s = add_slide("FUTURE DIRECTIONS", "What comes next",
              "Five directions that complete the system and address the limits found in Phase 1.")
fut = [
    ("Central system", "Vehicle database with status classification, real-time dashboard and alerts, dispatch by road-network distance, multi-node correlation."),
    ("Embedded deployment", "Port the node to a Jetson-class module; measure latency, memory and power; decide where UVMNet runs."),
    ("Real rain and night", "Test deraining and the weather router on real rain and night video; compare reading with and without restoration."),
    ("More data, better cameras", "Label more footage in fog, rain and at night; use higher-resolution cameras for two-wheeler and auto plates."),
    ("Reader improvements", "Multi-frame fusion in the live loop for low-vote vehicles; more state series; fix dropped repeated digits."),
]
pos = [(0.6, 1.85, 3.89), (4.72, 1.85, 3.89), (8.84, 1.85, 3.89), (0.6, 4.35, 5.95), (6.78, 4.35, 5.95)]
for i, ((x, y, w), (h, b)) in enumerate(zip(pos, fut)):
    rect(s, x, y, w, 2.3, adj=0.04, name="Direction card")
    badge(s, x + 0.25, y + 0.2, i + 1, [A1, A2, A3, A5, A1][i])
    text(s, x + 0.95, y + 0.2, w - 1.15, 0.5, P(R(h, 18, True, TX, "h")), anchor="m", name="Direction heading")
    text(s, x + 0.25, y + 0.9, w - 0.5, 1.35, P(R(b, 15)), name="Direction text")

# ---------------------------------------------------------------- order: cover, objectives, problem, literature, design..., spec table after architecture
order = [0, 1, 3, 4, 5, 2, 6]  # cover, objectives, problem, literature, architecture, spec, camera node
rest = list(range(7, len(prs.slides)))
lst = prs.slides._sldIdLst
els = list(lst)
new = [els[i] for i in order + rest]
for e in els:
    lst.remove(e)
for e in new:
    lst.append(e)

prs.save(OUT)
print("saved", OUT, len(prs.slides), "slides")
