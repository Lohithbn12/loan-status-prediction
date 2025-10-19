# src/evaluate.py
import argparse, os, json, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score, classification_report

def save_confusion_matrix(model, X_test, y_test, out="reports/figures/confusion_matrix.png"):
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); plt.savefig(out, dpi=160); plt.close()

def save_roc(model, X_test, y_test, out="reports/figures/roc_curve.png"):
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); plt.savefig(out, dpi=160); plt.close()

def save_feature_importance(model, feature_names, out="reports/figures/feature_importance.png"):
    # If the final estimator exposes feature_importances_ after preprocessing
    try:
        clf = model.base_estimator.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
            idx = np.argsort(imp)[::-1][:20]
            plt.barh(np.array(feature_names)[idx][::-1], imp[idx][::-1])
            plt.tight_layout(); os.makedirs(os.path.dirname(out), exist_ok=True); plt.savefig(out, dpi=160); plt.close()
    except Exception:
        pass

def dump_metrics(model, X_test, y_test, out="reports/metrics.json"):
    y_pred = model.predict(X_test)
    y_prob = getattr(model, "predict_proba", lambda X: None)(X_test)
    auc = roc_auc_score(y_test, y_prob[:,1]) if y_prob is not None else None
    rep = classification_report(y_test, y_pred, output_dict=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"roc_auc": auc, "report": rep}, open(out,"w"), indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--model", default="models/best_model.pkl")
    ap.add_argument("--plots", default="reports/figures")
    ap.add_argument("--metrics", default="reports/metrics.json")
    args = ap.parse_args()

    test = pd.read_csv(os.path.join(args.data, "test.csv"))
    y_test = test["Loan_Status"]
    X_test = test.drop(columns=["Loan_Status"])

    mdl = joblib.load(args.model)

    # Plots + metrics
    save_confusion_matrix(mdl, X_test, y_test, out=os.path.join(args.plots, "confusion_matrix.png"))
    save_roc(mdl, X_test, y_test, out=os.path.join(args.plots, "roc_curve.png"))
    # best-effort feature importance name mapping
    dump_metrics(mdl, X_test, y_test, out=args.metrics)

if __name__ == "__main__":
    main()
