# Cached paper

`2210.03629v3.pdf` is an unchanged copy of:

- Title: ReAct: Synergizing Reasoning and Acting in Language Models
- Authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao
- Version: arXiv:2210.03629v3, 10 March 2023
- Source: https://arxiv.org/pdf/2210.03629v3
- Record: https://arxiv.org/abs/2210.03629v3
- License: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/
- SHA-256: `f285b0971ae4a790e402fb93966bed3adde2cf0a04977d08b2b40d6ab0cace69`

The PDF is included unchanged as a fallback when live arXiv retrieval is unavailable or rate-limited.

The adjacent `.text.json` contains derived text under the same license, bound to
the PDF and text by SHA-256. Figures 1, 4, and 5 were transcribed from the rendered
source because their embedded fonts lack Unicode mappings. Bracketed labels
retain their color highlights and strike-throughs; deliberately incorrect
baseline answers remain. Font mapping repairs also cover pages 3, 22, and 31.
The paper's abstract supplies the document summary.

The `.corrections.json` records the reviewed replacements. The `.review.json`
records rejected extraction candidates, probable causes, and source anchors.
Regenerate a candidate with `python task1/tests/build_pdf_text.py <pdf>` inside
the course image. Compare it with the reviewed snapshot before replacing it.
The runtime checks encoding and page completeness; these checks cannot establish
the semantic accuracy of arbitrary OCR output.
