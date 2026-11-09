PY ?= python

.PHONY: all data batch forecast charts stream test clean

all: data batch forecast charts

data:
	$(PY) src/generate_data.py

batch:
	$(PY) src/batch_pipeline.py

forecast:
	$(PY) src/forecast_model.py

charts:
	$(PY) src/make_charts.py

stream:
	rm -rf data/stream
	$(PY) src/stream_simulator.py --batches 16 --interval 1.5 & \
	$(PY) src/streaming_monitor.py --duration 50 --fresh

test:
	$(PY) -m pytest -q tests

clean:
	rm -rf data
