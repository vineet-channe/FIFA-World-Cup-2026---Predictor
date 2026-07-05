.PHONY: api web install install-web catchup pipeline scheduler rebuild-groups rebuild-groups-dry verify-snapshots snapshot-check

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

rebuild-groups:
	source worldcup/bin/activate && \
	python scripts/rebuild_post_group_stage.py

rebuild-groups-dry:
	source worldcup/bin/activate && \
	python scripts/rebuild_post_group_stage.py --dry-run

verify-snapshots:
	source worldcup/bin/activate && \
	python scripts/verify_snapshots.py

snapshot-check:
	source worldcup/bin/activate && python scripts/save_snapshot.py --check
