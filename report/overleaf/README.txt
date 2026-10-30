WEATHER ADAPTIVE ANPR SYSTEM - project report (LaTeX, Overleaf-ready)

HOW TO USE IN OVERLEAF
1. Overleaf -> New Project -> Upload Project -> choose the .zip.
2. Menu -> Compiler: pdfLaTeX (default).  Main document: main.tex.
3. Press Recompile (Overleaf runs pdflatex + bibtex automatically).

WHERE TO EDIT
  main.tex                       packages, title/date macros (ReportTitle, ReportDate, AcademicYear, DeclarationDate)
  chapters/00_frontmatter.tex    cover, certificate (guide/coordinators/HOD), declaration, abstract,
                                 acknowledgement, abbreviations
  chapters/01_introduction.tex   Chapter 1  Introduction
  chapters/02_literature.tex     Chapter 2  Literature Review
  chapters/03_system_design.tex  Chapter 3  System Design (TikZ block diagrams)
  chapters/04_phase1.tex         Chapter 4  Phase 1: YOLOv11, restoration (DCP/CLAHE, UVMNet, physics-guided
                                 dehaze, Ponmudi real-fog study, deraining)
  chapters/04b_recognition_video.tex   Chapter 4 (continued): LPRNet, real-time pipeline, video tests,
                                 foggy-video pipeline, frame fusion, browser interface
  chapters/05_phase2_plan.tex    Chapter 5  Phase 2 plan
  chapters/06_conclusion.tex     Chapter 6  Conclusion and Future Work
  chapters/07_appendix.tex       Appendix: datasets, software, work log
  references.bib                 references (IEEEtran numeric style)
  figures/                       every picture used in the report

TO CHECK BEFORE SUBMISSION (search for "TODO" in chapters/00_frontmatter.tex)
  - Guide designation (Prof. Usha Anand) and the names of the project coordinators / HOD on the certificate.
  - Declaration date (DeclarationDate in main.tex).
  - Foggy-video plate readings (Table "Vehicles found in the four foggy clips") have not been checked against
    ground truth for all clips.
  - Deraining: Section 4.2.4 reports both PW-FNet (interim report) and the A5 network (deraining document);
    keep both or remove one so that it matches what the team presents.
  - Check the training numbers marked "should be verified" in the deraining document against the final log.
