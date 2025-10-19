# src/features.py
import argparse, os, pandas as pd, numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

def build_preprocessor(df):
    cat_cols = [c for c in df.columns if df[c].dtype=="object" and c!="Loan_Status"]
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

    num_pipe = SimpleImputer(strategy="median")
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    meta = {"num_cols": num_cols, "cat_cols": cat_cols}
    return pre, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/processed")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    train_path = os.path.join(args.in_dir, "train.csv")
    df = pd.read_csv(train_path)
    y = df["Loan_Status"] if "Loan_Status" in df.columns else None
    X = df.drop(columns=["Loan_Status"]) if "Loan_Status" in df.columns else df

    pre, meta = build_preprocessor(df)
    pre.fit(X)
    os.makedirs(args.out, exist_ok=True)
    joblib.dump({"pre": pre, "meta": meta}, os.path.join(args.out, "preprocessor.joblib"))

if __name__ == "__main__":
    main()
