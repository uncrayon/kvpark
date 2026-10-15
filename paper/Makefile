PAPER = main

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex refs.bib $(wildcard tables/*.tex figures/*)
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

# words in the body (abstract to the end of the conclusion), the competition's limit is 3,000
words:
	@detex $(PAPER).tex 2>/dev/null | wc -w || echo "detex not installed"

clean:
	latexmk -C $(PAPER).tex

.PHONY: all words clean
