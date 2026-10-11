PY ?= python
IMAGE ?= hotelmon:latest

.PHONY: setup data tune train replay monitor drift-perf simulate results latency readme test lint docker-build docker-run

setup:
	$(PY) -m pip install -r requirements.txt -e .

data:
	$(PY) -c "from hotelmon import data; print(data.download())"

tune:
	$(PY) scripts/tune_lightgbm.py

train:
	$(PY) scripts/train.py

replay:
	$(PY) scripts/replay.py

monitor:
	$(PY) scripts/monitor.py

drift-perf:
	$(PY) scripts/drift_vs_performance.py

simulate:
	$(PY) scripts/simulate_detectors.py

# Everything except latency (which needs the Docker container running).
results: data tune train replay monitor drift-perf simulate readme

readme:
	$(PY) scripts/render_readme.py

test:
	$(PY) -m pytest

lint:
	ruff check .
	ruff format --check .

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE)

latency:
	$(PY) scripts/load_test.py --url http://localhost:8000 --label docker
