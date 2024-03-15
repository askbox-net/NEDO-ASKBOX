# Pandocでthirds-solution-report.mdを./thirds-solution-report.pdfに変換する

```bash
pandoc -s -f markdown-implicit_figures -t beamer -H beamer.tex --pdf-engine xelatex -V colorlinks=true -V theme=Copenhagen -V colortheme=seahorse -V classopton=seahorse thirds-solution-report.md -o thirds-solution-report.pdf
```

# or
```bash
pandoc -d defaults.yaml
```
