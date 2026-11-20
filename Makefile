# Три цели прогона. Разделение здесь — не workflow-файл, а свойство окружения:
# `fast` обязана проходить там, где необязательных зигзаг-бэкендов нет.
#
# CI (.github/workflows/tests.yml) вызывает `fast` и только её, в окружении без
# [zigzag]; `slow` и `zigzag` остаются локальными целями.

PYTHON := .venv/bin/python

.PHONY: fast slow zigzag all install install-zigzag results report-zigzag

## Быстрый прогон: десяток секунд, без топологии и без бэкендов.
fast:
	$(PYTHON) -m pytest -q -m "not slow and not slow_zigzag"

## Топологические проверки: около десяти минут, ориентир бюджета 15.
slow:
	$(PYTHON) -m pytest -m slow --durations=15

## Зигзаг: оракул и бэкенды. Требует `make install-zigzag`.
zigzag:
	$(PYTHON) -m pytest -m slow_zigzag --durations=15

all: fast slow zigzag

## Артефакт Результата 2. Производит сборка, а не pytest.
results:
	$(PYTHON) -m tdadyn.results --config config.yaml --results

## Отчёт Результата 2: собрать, ВЫПОЛНИТЬ и выгрузить в HTML. Без выполнения утверждение
## «каждое число — вывод исполненной ячейки» не проверяемо, а отчёт Задачи 5 именно в таком
## состоянии: его html собран nbconvert'ом, которого не было в зависимостях.
report-zigzag: results
	$(PYTHON) notebooks/build_zigzag_report.py
	cd notebooks && ../$(PYTHON) -c "import nbclient, nbformat; \
	  nb = nbformat.read('zigzag_report.ipynb', as_version=4); \
	  nbclient.NotebookClient(nb, timeout=600, kernel_name='python3').execute(); \
	  nbformat.write(nb, 'zigzag_report.ipynb'); \
	  print('выполнено:', sum(1 for c in nb.cells if c.cell_type == 'code'), 'ячеек кода')"
	$(PYTHON) -m nbconvert --to html --embed-images \
	  --output zigzag_report.html notebooks/zigzag_report.ipynb

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-zigzag:
	$(PYTHON) -m pip install -e ".[dev,zigzag]"
