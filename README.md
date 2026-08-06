# nimb-ou.github.io

Portfolio site for two machine learning projects. Static HTML, no build step, no
dependencies — the charts are hand-rolled SVG so the page cannot break because
someone else's CDN did.

| | |
|---|---|
| `index.html` | Landing page |
| `credit.html` | Loan Underwriting Copilot — threshold explorer, applicant browser, fairness |
| `power.html` | Power Price Alpha — day explorer, intervals, battery backtest |
| `course.html` | All 31 lessons across both projects |
| `data/*.json` | Exported from the project repos — **never edited by hand** |

## The data is generated, not written

Every number on this site is exported from the projects themselves:

```bash
cd ../loan-underwriting-copilot && python tools/export_site_data.py --out ../portfolio-site/data/credit.json
cd ../power-price-alpha      && python tools/export_site_data.py --out ../portfolio-site/data/power.json
```

Those scripts read `reports/metrics.json` and the trained artifacts, so a claim
on this site cannot drift from the model that produced it. If a figure here looks
wrong, re-run the exporter rather than editing the JSON.

## Local preview

```bash
python3 -m http.server 4321
```

Opening the files directly with `file://` will not work — the browser blocks
`fetch` for local files, so the data never loads. The page detects this and says
so rather than showing empty charts.
