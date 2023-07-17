# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
BUILDDIR      = build

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source
# the i18n builder cannot share the environment and doctrees with the others
I18NSPHINXOPTS  = $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) source

.PHONY: help clean build publish

help:
	@echo "Build iohub HTML documentation."
	@echo
	@echo "Usage: make COMMAND"
	@echo
	@echo "Commands:"
	@echo "build\tBuild from current source code"
	@echo "publish\tBuild multiple versions for publishing"
	@echo "clean\tRemove all built files"
	@echo "help\tShow this message and quit"

clean:
	-rm -rf $(BUILDDIR)/*
	-rm -rf source/auto_examples/*

build:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

publish:
	sphinx-multiversion source build/html
	cp gh-pages-redirect.html build/html/index.html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."
