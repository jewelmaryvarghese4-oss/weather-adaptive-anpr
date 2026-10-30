"""Final-review study guide (questions and answers) as a Word document."""
from docx import Document
from docx.shared import Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT = r"C:\projects\jewwww\documents\Final_Review_Questions_and_Answers.docx"
TEAL = RGBColor(0x0F, 0x76, 0x6E)
ORANGE = RGBColor(0xB4, 0x53, 0x09)
GREY = RGBColor(0x4A, 0x62, 0x66)

d = Document()
sec = d.sections[0]
sec.page_width, sec.page_height = Cm(21), Cm(29.7)
sec.left_margin = sec.right_margin = Cm(2.2)
sec.top_margin = sec.bottom_margin = Cm(2.0)

st = d.styles["Normal"]
st.font.name = "Calibri"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
st.paragraph_format.space_after = Pt(4)
for name, size, color in (("Heading 1", 18, TEAL), ("Heading 2", 14, TEAL), ("Heading 3", 12, ORANGE)):
    h = d.styles[name]
    h.font.name = "Cambria"; h.font.size = Pt(size); h.font.bold = True; h.font.color.rgb = color
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Cambria")
    h.paragraph_format.space_before = Pt(14 if name != "Heading 3" else 8)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True


def footer_page_numbers():
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("Weather Adaptive ANPR \u00b7 Final review study guide \u00b7 page ")
    r.font.size = Pt(9); r.font.color.rgb = GREY
    for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        run = p.add_run(); run.font.size = Pt(9); run.font.color.rgb = GREY
        if t:
            fc = OxmlElement("w:fldChar"); fc.set(qn("w:fldCharType"), t); run._r.append(fc)
        else:
            it = OxmlElement("w:instrText"); it.set(qn("xml:space"), "preserve"); it.text = txt; run._r.append(it)


footer_page_numbers()


def para(text="", bold=False, italic=False, color=None, size=None, after=None, left=None):
    p = d.add_paragraph()
    if left is not None:
        p.paragraph_format.left_indent = Cm(left)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    add_runs(p, text, bold, italic, color, size)
    return p


def add_runs(p, text, bold=False, italic=False, color=None, size=None):
    """**bold** segments inside text are rendered bold."""
    parts = text.split("**")
    for i, seg in enumerate(parts):
        if not seg:
            continue
        r = p.add_run(seg)
        r.bold = bold or (i % 2 == 1)
        r.italic = italic
        if color is not None:
            r.font.color.rgb = color
        if size:
            r.font.size = Pt(size)


def bullet(text, level=0):
    p = d.add_paragraph(style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.9 + 0.7 * level)
    p.paragraph_format.space_after = Pt(2)
    add_runs(p, text)
    return p


def numbered(i, text):
    p = d.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.1)
    p.paragraph_format.first_line_indent = Cm(-0.6)
    p.paragraph_format.space_after = Pt(2)
    add_runs(p, "%d.  %s" % (i, text))


def qa(n, q, a):
    p = d.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    add_runs(p, "Q%d.  %s" % (n, q), bold=True, color=TEAL)
    for line in a:
        if line.startswith("- "):
            bullet(line[2:])
        else:
            para(line, left=0.9)


# ------------------------------------------------------------------ title
t = d.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.LEFT
r = t.add_run("Weather Adaptive ANPR System"); r.font.name = "Cambria"; r.font.size = Pt(26); r.bold = True; r.font.color.rgb = RGBColor(0x0B, 0x2B, 0x2E)
t2 = d.add_paragraph(); r = t2.add_run("Final review: study guide with likely questions and answers"); r.font.name = "Cambria"; r.font.size = Pt(15); r.font.color.rgb = TEAL
para("Auxliya Mariya \u00b7 Jewel Mary Varghese \u00b7 Megha T S \u00b7 Silpa C R \u00b7 Guide: Prof. Usha Anand", color=GREY, after=10)

para("How to use this guide. Read the **Say this** lines aloud once; they are the short answers to give. The explanation under each is what to fall back on "
     "if the examiner asks \u201cwhy\u201d or \u201chow\u201d. Where a result has a limit, say the limit yourself; it is stated in each answer below. "
     "If you do not remember an exact figure, say that it is in the report and offer to show it.", italic=True, after=8)

d.add_heading("The project in 20 seconds", level=2)
para("A camera frame goes through weather restoration, then YOLOv11 finds the plate, then LPRNet reads it, then voting over frames confirms it. "
     "UVMNet is the fog-removal part. The main finding is that cleaner pictures help the viewer more than the reader; frame voting and a reader trained "
     "for bad weather are what raise plate accuracy.")

# ------------------------------------------------------------------ Part 1: core questions
d.add_heading("Part 1. The core questions on UVM and the physics-guided dehaze", level=1)

d.add_heading("Q1. What is a UVM model?", level=3)
para("**Say this:** UVMNet is a deep-learning dehazing network. It is a U-shaped encoder\u2013decoder with Mamba state-space blocks inside it. "
     "It takes a foggy image and outputs a fog-free one. Ours has 116.2 million parameters.")
bullet("**U-shape:** the image is shrunk step by step (the encoder) to capture the whole scene, then enlarged (the decoder) to rebuild detail. "
       "Skip connections pass fine detail across, so edges are not lost.")
bullet("**Mamba:** a state-space model that reads an image as a long sequence of pixels. It keeps a memory (the \u201cstate\u201d) and decides what to keep from the input. "
       "Its cost grows linearly with image size, whereas a transformer\u2019s grows quadratically.")
bullet("**Output:** the network predicts a correction that is added to the foggy input (output = network(x) + x). It makes no separate fog map.")

d.add_heading("Q2. What is the algorithm behind UVM?", level=3)
para("**Say this:** every encoder and decoder stage starts with a UVMB block. It runs three Mamba layers: one scans the pixels as a sequence, one scans across "
     "channels, and one produces attention weights. The weights multiply the channel-scan result, and convolutions then refine it.")
para("Step by step inside a UVMB block:", after=2)
for i, s in enumerate([
    "A small convolution adds local detail (with a residual connection).",
    "The features become a sequence and are layer-normalised.",
    "Mamba 1 runs along the pixel sequence, which captures global context.",
    "Mamba 3 runs across channels, using the flattened spatial size as its dimension.",
    "Mamba 2 followed by a softmax gives attention weights; the result is attention \u00d7 (channel-scan output).",
], 1):
    numbered(i, s)
para("Training:", after=2)
bullet("The output is compared to the clear image with Charbonnier loss; we added gradient and residual regularisation terms.")
bullet("Stage A used 8,925 traffic pairs and reached 32.96 dB. Stage C added RESIDE-OTS, 1,001 Foggy-Hazy plate pairs and NH-HAZE, and reached 27.30 dB.")
bullet("We shrank the block\u2019s internal size to 32 so the model fits on our 8 GB GPU.")

d.add_heading("Q3. Why was UVM used here?", level=3)
para("**Say this:** fog thickness depends on distance and varies across the image, so the model needs global context. Mamba gives that context at linear cost on large photos. "
     "DCP failed on dense fog, CLAHE has no haze model, and our first transformer network smoothed away plate edges.")
bullet("Numbers to quote: DCP turned dense fog white; the transformer\u2019s OCR confidence was 0.104 against 0.326 for the hazy input.")
bullet("**Honest limit:** the Mamba kernel is CUDA-only. That is why the node targets a Jetson module, and its latency there is a Phase 2 test.")

d.add_heading("Q4. What is the algorithm behind the other models?", level=3)
for s in [
    "**YOLOv11n:** one network predicts plate boxes and confidence in one pass. Trained for 100 epochs on 15,000+ images; mAP@50 99.2 %.",
    "**LPRNet:** a CNN reads the plate crop (200\u00d724) and outputs character probabilities per column. **CTC loss** trains it without character boxes by summing over all alignments, "
    "so repeated characters need a blank between them (this is why 888 becomes 88). We add format-constrained decoding: letters and digits in the right slots (KL 01 AB 1234), picked using the network\u2019s own probabilities.",
    "**DCP (Dark Channel Prior):** in clear outdoor patches one colour channel is near zero. Fog lifts that \u201cdark channel\u201d, so it estimates fog and atmospheric light and inverts the haze equation. "
    "No training needed, but it fails on dense, uniform fog.",
    "**CLAHE:** equalises contrast tile by tile with a clip limit so noise is not amplified. Fast (about 1.3 ms); has no haze model.",
    "**Transformer U-Net:** window attention in the bottleneck, trained on synthetic pairs. Best-looking car, but plate edges were lost.",
    "**Deraining (A5):** a small U-Net with about 1.72 million parameters trained on clean images with synthetic rain; loss = Charbonnier + 0.2 \u00d7 (1 \u2212 SSIM) with extra weight on the plate region. "
    "Reached 29.19 dB on synthetic rain. NeRD-Rain is the baseline.",
    "**Tracking and voting:** each plate is followed across frames and its readings voted; confirmed after 5 or more agreeing reads.",
    "**Frame fusion:** crops of the same plate are aligned with ECC, upsampled, averaged, then read.",
]:
    bullet(s)

d.add_heading("Q5. Why go for the physics-guided \u201cstrong + exposure\u201d?", level=3)
para("**Say this:** the network is good at measuring fog but sometimes invents texture or leaves seams. So we use it only to measure the fog thickness, "
     "then remove the fog from the real photo with the fog equation. \u201cStrong\u201d removes more fog, and \u201cexposure\u201d stops the picture getting dark.")

d.add_heading("Q6. What is the concept behind it?", level=3)
para("The fog law is **I = J\u00b7t + A\u00b7(1 \u2212 t)**.", after=2)
bullet("**I** is the foggy image we have. **J** is the clear scene we want.")
bullet("**t** is transmission, how much of the scene survives the fog (1 is clear, 0 is fully fogged). **A** is the veil colour, the brightness of the fog itself.")
para("Our steps:", after=2)
for i, s in enumerate([
    "The network\u2019s output is used to measure t.",
    "A comes from the brightest 0.1 % of the dark channel.",
    "We invert: **J = (I \u2212 A) / t^\u03b3 + A**. \u03b3 greater than 1 is \u201cstrong\u201d: a smaller t means more fog is removed. We used \u03b3 = 1.25. We also put a floor on t and cap the detail gain at 1.5.",
    "Removing the veil darkens the image, so exposure matching brings the mean brightness back to about 0.46.",
], 1):
    numbered(i, s)
para("Why this works: the real photo\u2019s pixels are kept, so there are no ghost trees and no tile seams.", left=0.9)
para("**Honest result:** the pictures are the cleanest. But the installed reader read 49 of 85 plates from them, against 47\u201351 for a weather-trained reader on the untouched photos. "
     "That is inside the test noise of about \u00b13 plates. So the limit is the information in the photo, not the dehazer.", left=0.9)

d.add_heading("Q7. What is the clip size?", level=3)
para("\u201cClip size\u201d is ambiguous, so first ask the examiner: \u201cDo you mean the CLAHE clip limit?\u201d That is the likely one.")
bullet("**CLAHE clip limit = 3.0**, with an 8\u00d78 tile grid. It caps how much each tile\u2019s histogram can be stretched, so noise is not amplified.")
para("If they meant a different size:", after=2, left=0.9)
bullet("**Training crop:** 384\u00d7384 for UVMNet, 256\u00d7256 for deraining.")
bullet("**UVMNet internal size:** 32 (the block\u2019s fixed internal resolution).")
bullet("**Inference tile:** 1536 px with 128 px overlap, blended to avoid seams.")
bullet("**Reader input:** 200\u00d724. **Detector input:** 1280 px.")

# ------------------------------------------------------------------ Part 2
d.add_heading("Part 2. More likely questions", level=1)
n = 7
sections = [
 ("A. Project and system", [
  ("What problem does your project solve?", ["Plate recognition fails in fog, rain and low contrast. We built a camera-node pipeline that restores the frame, finds the plate, reads it and confirms it by voting. The central system (database, alerts, dispatch, correlation) is Phase 2."]),
  ("What is the difference between Phase 1 and Phase 2?", ["Phase 1 is the vision core, from camera frame to confirmed plate, tested on a laptop GPU. Phase 2 is vehicle verification, alerts and dispatch, multi-node correlation and Jetson deployment."]),
  ("Why two layers, a node and a central system?", ["The node reads plates locally and sends only the number, a crop, the camera ID and the time, so network load stays small. The node also keeps working if the link is lost."]),
  ("Why Jetson and not Raspberry Pi?", ["The Mamba kernel in UVMNet is CUDA-only, so it will not run on a Pi. Whether it meets the latency budget on the Jetson module is a Phase 2 test."]),
  ("What is the pipeline order?", ["Frame \u2192 weather check \u2192 restoration \u2192 YOLOv11 plate detection \u2192 crop and flatten \u2192 LPRNet with format decode \u2192 track and vote \u2192 confirmed plate. We support restoring the whole frame first, or detecting first and restoring only the crop."]),
 ]),
 ("B. Detection", [
  ("Why YOLOv11n?", ["It finds boxes in one pass, so it is fast, and the nano size fits embedded hardware. We got 99.2 % mAP@50."]),
  ("What do precision, recall and mAP mean?", ["Precision is the share of detections that are real plates (98.5 %). Recall is the share of real plates found (99.0 %). mAP@50 is the average precision when a box counts as correct if it overlaps the true box by at least 50 %. mAP@50\u201395 averages stricter overlaps; ours is 85.6 %."]),
  ("If detection is 99 %, why is plate reading only about 50 %?", ["Detecting a rectangle is easy. Reading blurred, small, foggy characters is hard, and the limit is the information in the photo."]),
  ("Why run the detector at 1280 pixels?", ["So small plates in 1080p and 4K frames are not lost."]),
 ]),
 ("C. Restoration in general", [
  ("Why restore at all if the reader sees fog?", ["Restoration makes the picture usable for the operator. Our honest finding is that, once the reader is trained for weather, restoration gives little extra plate reading."]),
  ("Why did you choose CLAHE first?", ["It is the fastest (about 1.3 ms against 17 ms for DCP) and never fails badly. It still cannot restore detail that was never captured."]),
  ("What is DCP and why did it fail?", ["Dark Channel Prior assumes that in clear outdoor patches one colour channel is near zero. Dense, uniform fog covers the whole frame, so the assumption breaks. With gentle settings it did nothing, and with strong settings the frame turned white."]),
  ("What is the atmospheric scattering model?", ["I = J\u00b7t + A\u00b7(1\u2212t). I is the foggy image, J the clear scene, t how much of the scene survives, and A the veil colour. Dehazing means recovering J."]),
  ("Why did the transformer dehazer fail?", ["It was trained only on synthetic pairs from plate crops. It gave a clean-looking car but smoothed away plate edges, and its OCR confidence was 0.104 against 0.326 for the hazy input."]),
 ]),
 ("D. UVMNet and physics-guided dehaze", [
  ("What is Mamba, in simple words?", ["A sequence model with a memory that decides, for each input, what to keep or forget. Its cost grows linearly with length, unlike a transformer\u2019s quadratic cost."]),
  ("Why did you train UVMNet in stages?", ["Each stage fixed a problem from the previous one. Stage A on traffic data gave a colour cast on our photos. Stage B on plate crops fixed brightness (46 % to 70 %). Stage C added more datasets and reached 27.30 dB."]),
  ("What is PSNR, and why is it not enough?", ["PSNR measures pixel-level similarity to the clear image, in dB. A model can score high PSNR yet erase plate edges, so we also read the plates."]),
  ("What is the guided filter and why use it?", ["An edge-preserving smoother. We apply it after UVMNet because raw network output read worse. On the newest model it hurt reading, so it is not always used."]),
  ("Why not use UVMNet\u2019s output directly?", ["It sometimes leaves tile seams and ghost trees and shifts colours. The physics step keeps the real photo\u2019s pixels and removes only the fog."]),
  ("What does \u201cstrong\u201d mean?", ["A higher strength exponent \u03b3 (we used 1.25). It lowers the effective transmission, so more fog is removed."]),
  ("What does the exposure match do?", ["Removing the veil darkens the picture. Exposure matching brings the mean brightness back to about 0.46."]),
  ("Which model looked best?", ["The physics-guided output on a 50/50 blend of the GoPro hybrid and the new model, strong with exposure match. Its haze score was 0.261 and its grain 0.020."]),
  ("What is the depth-fog model?", ["It was trained on synthetic fog made from Depth Anything V2 depth maps, so fog is thicker with distance. It gained about 4 dB on its own test but nothing on real photos."]),
 ]),
 ("E. Deraining", [
  ("What is your deraining model?", ["A custom U-Net (A5) with about 1.72 million parameters, trained on clean RodoSol-ALPR images with synthetic rain. It reached 29.19 dB against 22.17 dB unprocessed."]),
  ("Does deraining improve plate reading?", ["Not shown yet. On real rain it reduces some streaks, but dense rain, glare and spray remain. That is a Phase 2 evaluation."]),
  ("Why is there a synthetic-to-real gap?", ["Synthetic streaks are simpler than real rain, which also has splash, wet-surface sheen and lens drops. PW-FNet scored 36.46 dB on synthetic plates but gave almost no real-rain gain (23.64 against 23.60 dB)."]),
 ]),
 ("F. Plate reader", [
  ("Why did EasyOCR fail?", ["It is a general scene-text reader. It read only 16.5\u201325.2 % of fog plates exactly and confused characters such as 1/I and 7/Z."]),
  ("What is CTC loss?", ["It trains a sequence reader without per-character boxes by summing over all alignments between the output columns and the target string. A blank symbol separates repeated characters, which is why 888 can become 88."]),
  ("Why is the input only 200\u00d724?", ["Plates are wide and thin, and this keeps the network small and fast. Two-line plates are flattened into one row first."]),
  ("What is format-constrained decoding?", ["Indian plates follow two letters, two digits, one or two letters, then four digits. The decoder picks, for each slot, the best character that slot allows, using the network\u2019s own probabilities."]),
  ("What helped the reader most?", ["Real data. Synthetic pre-training took the character error from 0.70 to 0.32, and adding CCPD and mixing brought it to 0.249. Labelled real-video crops raised unseen-video plates from 38 to 41 of 51."]),
  ("Why does it still fail on some plates?", ["Repeated digits (448 for 4448), series from other states, and small, blurred or faded plates. The two-wheeler and auto plates are only 22\u201339 pixels wide."]),
 ]),
 ("G. Real time, voting and fusion", [
  ("How do you confirm a plate?", ["Each plate is tracked and its readings voted. A plate is confirmed after 5 or more reads agree, or announced as it leaves if the reading is well formed and most reads agree."]),
  ("How fast is it?", ["Up to 55 fps at 1080p, and about 20\u201330 fps at 4K. We read the first ten reads of a plate every frame, then every third frame, and skip frames if we fall behind."]),
  ("What is multi-frame fusion?", ["We align several crops of one plate with ECC, upsample them, average them and read the result. On the Ponmudi tracks a single frame read 1 of 36 and a vote 9. Fusion gave 8\u201310, about the same as voting, so we call it promising only."]),
 ]),
 ("H. Evaluation and honesty", [
  ("How do you know your results are right?", ["We test on held-out data that was never trained on: 16 Kerala plates, six unseen videos (51 plates) and 62 held-out Rajasthan crops. The Ponmudi set has 85 plates verified by eye, with noise of about \u00b13 plates."]),
  ("What is the biggest limitation?", ["The information in the photo. A plate that fog and grain have blurred cannot be recovered from a single frame by any dehazer we built."]),
  ("What would you do next?", ["Build the database, alerts and correlation, test on the Jetson, evaluate deraining on real rain and night video, and collect more labelled footage and higher-resolution cameras."]),
  ("If asked for a number you do not remember", ["Say: \u201cI don\u2019t have that exact figure in my head, but it is in the report and I can show you.\u201d Do not guess."]),
 ]),
]
for title, items in sections:
    d.add_heading(title, level=2)
    for q, a in items:
        n += 1
        qa(n, q, a)

# ------------------------------------------------------------------ cheat sheet
d.add_heading("Part 3. Numbers to remember", level=1)
rows = [
    ("Detector (YOLOv11n)", "precision 98.5 %, recall 99.0 %, mAP@50 99.2 %, mAP@50\u201395 85.6 %; 15,000+ images, 100 epochs"),
    ("EasyOCR in weather", "16.5\u201325.2 % exact in fog; 5.9\u201311.0 % in rain"),
    ("LPRNet character error", "0.70 (scratch) \u2192 0.32 (synthetic + real) \u2192 0.249 (+ CCPD and mixing)"),
    ("Reader on unseen videos", "38 \u2192 41 of 51 plates (six test videos); 19 \u2192 22 of 27 (video 4)"),
    ("Rajasthan held-out crops", "48 \u2192 56 of 62 exact; restored photos 205 vs hazy 195 of 248"),
    ("UVMNet", "116.2 M parameters; 32.96 dB (traffic), 27.30 dB (multi-dataset); tiles 1536 px, overlap 128"),
    ("Physics-guided", "\u03b3 = 1.25; A from top 0.1 % of dark channel (clipped 0.55\u20130.97); detail gain cap 1.5; exposure target 0.46"),
    ("Ponmudi test", "60 photos, 85 plates; installed reader 40 \u2192 44\u201349 after dehaze; weather-trained reader 47\u201351 untouched; noise \u00b13"),
    ("Deraining (A5)", "1.72 M parameters; 29.19 dB vs 22.17 dB unprocessed; NeRD-Rain 20.98 dB"),
    ("Speed", "up to 55 fps at 1080p; about 20\u201330 fps at 4K; 10 of 12 plates exact on unseen 4K video"),
    ("Frame fusion", "36 readings: single frame 1, vote 9, fused 8\u201310"),
    ("CLAHE", "clip limit 3.0, 8\u00d78 tiles; about 1.3 ms vs DCP 17 ms"),
]
tbl = d.add_table(rows=0, cols=2)
tbl.style = "Table Grid"
for a, b in rows:
    cells = tbl.add_row().cells
    cells[0].width = Cm(4.6); cells[1].width = Cm(12.0)
    cells[0].paragraphs[0].add_run(a).bold = True
    cells[1].paragraphs[0].add_run(b)
    for c in cells:
        for p in c.paragraphs:
            p.paragraph_format.space_after = Pt(2)
            for r in p.runs:
                r.font.size = Pt(10.5)
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "EAF3F1")
    cells[0]._tc.get_or_add_tcPr().append(shd)

d.save(OUT)
print("saved", OUT, "questions:", n)
