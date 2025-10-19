# src/data.py
import argparse, os, pandas as pd
from sklearn.model_selection import train_test_split

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.input)
    # Basic cleaning: strip whitespace in column names
    df.columns = [c.strip() for c in df.columns]
    # Save full clean copy
    df.to_csv(os.path.join(args.out, "clean.csv"), index=False)

    # Train/Test split (stratify if target exists)
    target = "Loan_Status" if "Loan_Status" in df.columns else None
    if target:
        train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42, stratify=df[target])
    else:
        n = int(len(df)*(1-args.test_size))
        train_df, test_df = df.iloc[:n].copy(), df.iloc[n:].copy()

    train_df.to_csv(os.path.join(args.out, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.out, "test.csv"), index=False)

if __name__ == "__main__":
    main()
