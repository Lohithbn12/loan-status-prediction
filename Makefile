.PHONY: data train evaluate lint

data:
	python -m src.data --input data/raw/loan_approval_dataset.csv --test-size 0.2 --out data/processed

train:
	python -m src.model --train --data data/processed --save models/best_model.pkl

evaluate:
	python -m src.evaluate --data data/processed --model models/best_model.pkl --plots reports/figures --metrics reports/metrics.json

lint:
	python -m pip install ruff && ruff check .
