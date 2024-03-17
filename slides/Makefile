PANDOC = pandoc
TARGET = thirds-solution-report.pdf
SRC = thirds-solution-report.md
BEAMER = beamer.tex

FROM = -f markdown-implicit_figures 
TO = -t beamer

FLAGS = -H $(BEAMER) \
	--pdf-engine xelatex \
	-V colorlinks=true \
	-V theme=Copenhagen \
	-V colortheme=seahorse \
	-V classopton=seahorse

$(TARGET): $(SRC)
	$(PANDOC) $(FROM) $(TO) $(FLAGS) $< -o $@

$(TARGET): $(SRC) $(BEAMER)

clean:
	rm -rf $(TARGET)

.PHONY: clean

