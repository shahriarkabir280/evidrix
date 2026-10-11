# Test fixtures

| File | Source | License |
|---|---|---|
| `glued-words-cid-fonts.pdf` | Page 8 of the Python 3.12 "Logging HOWTO" PDF (docs.python.org/3.12/archives/python-3.12-docs-pdf-a4.zip) | PSF-2.0 |

`glued-words-cid-fonts.pdf` is a regression fixture: its Type0 / Identity-H fonts make pypdf drop the spaces
between words ("Filtersarecoveredinmoredetail…"). Generated PDFs don't reproduce this, so the real page is kept.
