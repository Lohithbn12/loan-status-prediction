# src/model.py
import argparse, os, pandas as pd, numpy as np, joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

def train_model(X, y, pre, model_type="rf"):
    if model_type == "logreg":
        clf = LogisticRegression(max_iter=1000, n_jobs=None)
    else:
        clf = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe.fit(X, y)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--save", default="models/best_model.pkl")
    ap.add_argument("--model", choices=["logreg","rf"], default="rf")
    args = ap.parse_args()

    train = pd.read_csv(os.path.join(args.data, "train.csv"))
    test = pd.read_csv(os.path.join(args.data, "test.csv"))
    y_train, y_test = train["Loan_Status"], test["Loan_Status"]
    X_train, X_test = train.drop(columns=["Loan_Status"]), test.drop(columns=["Loan_Status"])

    pre_bundle = joblib.load(os.path.join(args.data, "preprocessor.joblib"))
    pre = pre_bundle["pre"]

    model = train_model(X_train, y_train, pre, args.model)
    # Calibrate
    cal = CalibratedClassifierCV(model, method="isotonic", cv=3)
    cal.fit(X_train, y_train)

    # Quick metrics
    y_pred = cal.predict(X_test)
    y_prob = cal.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Test Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f}")

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    joblib.dump(cal, args.save)

if __name__ == "__main__":
    main()
