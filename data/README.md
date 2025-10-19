# Data

- **Source:** Provided by user (replace with Kaggle/UCI link if public).
- **File:** `data/raw/loan_approval_dataset.csv`
- **Note:** Keep sensitive data out of the repo. Include only samples if redistribution is restricted.

## Quickstart
```bash
# Process and split data
python -m src.data --input data/raw/loan_approval_dataset.csv --test-size 0.2 --out data/processed
```
