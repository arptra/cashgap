.PHONY: setup dev test demo clean

setup:
	./scripts/setup.sh

dev:
	./scripts/dev.sh

test:
	./scripts/test.sh

demo:
	./scripts/demo.sh

clean:
	rm -rf frontend/dist backend/.pytest_cache backend/__pycache__ backend/app/__pycache__ backend/tests/__pycache__
	rm -rf data/raw/* data/normalized/* data/synthetic/* data/uploaded/*
	rm -rf artifacts/runs/* artifacts/models/* cache/* logs/*
	rm -f backend/cashgap.db backend/cashgap.db-shm backend/cashgap.db-wal
