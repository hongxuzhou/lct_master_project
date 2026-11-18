# Research proposal v4: web edition

The complete submitted proposal renders as one self-contained HTML page,
including the references and all five appendices. The original LaTeX sources,
wording, values, SBN sequences and timeline remain unchanged.

From this directory:

```sh
quarto preview index.qmd
quarto render index.qmd
```

`index.qmd` includes the six files in `sections/` and renders to `index.html`.
Section, figure and table numbers match the original proposal. Appendix
headings retain their original lettered numbers and have direct anchor links.

`references.bib` contains only the 40 cited entries, copied verbatim from
`../my_library.bib`. `references.csl` is Pandoc's bundled Chicago author-date
style; it does not reproduce `plainnat` typography exactly.

The SBN filter adds separator, error and comment highlighting without changing
the code text. Comparison panels stack on screens narrower than 900px. Wide
tables scroll within the article. The original TikZ diagram is redrawn as an
accessible SVG in `figures/paired-transformation.svg`; its labels and arrow
directions are preserved. The Siri image is copied from the original source.

The HTML embeds its styles and images and is ready for static hosting on
Posit Connect Cloud. This directory does not publish automatically.
