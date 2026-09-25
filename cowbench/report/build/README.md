# Report generator

Rebuilds `../Muse_Glimmer_CBVD5_test.docx` from the archived runs in `../../runs/`.
No number in the document is typed by hand.

```bash
npm install docx            # once, in this folder
python facts.py             # every figure, computed from runs/ -> facts.json
python charts.py            # chart_recall.png, chart_size.png, size_split.json
node build.js ../Muse_Glimmer_CBVD5_test.docx
```

`build.js` also needs `example_frame.jpg` (Figure 1: clip 371, t=5 s, one cow outlined),
rendered with `render.render(..., mode="marked", max_width=1280)`.
