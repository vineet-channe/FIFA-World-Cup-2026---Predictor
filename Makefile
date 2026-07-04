.PHONY: api web install install-web catchup pipeline scheduler

api:
	source worldcup/bin/activate && uvicorn src.api.main:app --reload --port 8000

web:
	cd web && npm run dev

install:
	pip install -r requirements.txt

install-web:
	cd web && npm install

catchup:
	source worldcup/bin/activate && python scripts/05_catchup.py

pipeline:
	source worldcup/bin/activate && python scripts/06_run_pipeline.py

scheduler:
	source worldcup/bin/activate && python scripts/07_start_scheduler.py
