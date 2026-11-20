# Three test targets. The split is not a workflow file but a property of the environment:
# `fast` must pass where the optional zigzag backends are absent.
#
# CI (.github/workflows/tests.yml) calls `fast` and only `fast`, in an environment without
# [zigzag]; `slow` and `zigzag` remain local targets.

PYTHON := .venv/bin/python

.PHONY: fast slow zigzag all install install-zigzag results report-zigzag

## Fast run: a few dozen seconds, no topology and no backends.
fast:
	$(PYTHON) -m pytest -q -m "not slow and not slow_zigzag"

## Topological checks: about ten minutes, budget guideline 15.
slow:
	$(PYTHON) -m pytest -m slow --durations=15

## Zigzag: oracle and backends. Requires `make install-zigzag`.
zigzag:
	$(PYTHON) -m pytest -m slow_zigzag --durations=15

all: fast slow zigzag

## The zigzag analysis artefact. Produced by the build, not by pytest.
results:
	$(PYTHON) -m tdadyn.results --config config.yaml --results

## The zigzag report: build, EXECUTE and export to HTML. Without execution the claim
## "every number is the output of an executed cell" cannot be checked.
report-zigzag: results
	$(PYTHON) notebooks/build_zigzag_report.py
	cd notebooks && ../$(PYTHON) -c "import nbclient, nbformat; \
	  nb = nbformat.read('zigzag_report.ipynb', as_version=4); \
	  nbclient.NotebookClient(nb, timeout=600, kernel_name='python3').execute(); \
	  nbformat.write(nb, 'zigzag_report.ipynb'); \
	  print('executed:', sum(1 for c in nb.cells if c.cell_type == 'code'), 'code cells')"
	$(PYTHON) -m nbconvert --to html --embed-images \
	  --output zigzag_report.html notebooks/zigzag_report.ipynb

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-zigzag:
	$(PYTHON) -m pip install -e ".[dev,zigzag]"
