# Три цели прогона. Разделение здесь — не workflow-файл, а свойство окружения:
# `fast` обязана проходить там, где необязательных зигзаг-бэкендов нет.
#
# CI появится вместе с remote и будет вызывать эти же цели; до тех пор «зелёный CI»
# остаётся открытым пунктом Результата 4, а не Z0 (docs/plan_task2.md, §Z0).

PYTHON := .venv/bin/python

.PHONY: fast slow zigzag all install install-zigzag

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

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-zigzag:
	$(PYTHON) -m pip install -e ".[dev,zigzag]"
