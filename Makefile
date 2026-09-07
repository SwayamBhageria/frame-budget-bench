PY ?= venv/bin/python

.PHONY: setup data results bench docs check test web all clean

setup:
	python3 -m venv venv && $(PY) -m pip install -q --upgrade pip && $(PY) -m pip install -q -r requirements.txt

data:
	$(PY) tools/fetch_data.py

results:
	$(PY) -m fbb results

bench:
	$(PY) -m fbb bench

docs:
	$(PY) tools/inject.py

check:
	$(PY) tools/inject.py --check

web:
	$(PY) tools/export_web.py

test:
	$(PY) -m pytest tests/ -q

all: results bench docs web test

clean:
	rm -rf results/*.json web/data.json __pycache__ fbb/__pycache__ .pytest_cache
