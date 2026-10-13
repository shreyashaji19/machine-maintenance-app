"""Train Logistic Regression, Decision Tree and Random Forest; keep the best.
Run from the project root:  python -m src.train
"""
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.preprocessing import FEATURES, load_data, split_xy

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

MODELS = {
    "Logistic Regression": (
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED))]),
        {"clf__C": [0.1, 1, 10, 100]}),
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
        {"max_depth": [5, 8, 12, None], "min_samples_leaf": [1, 5, 10]}),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=SEED, n_jobs=-1),
        {"max_depth": [10, None], "min_samples_leaf": [1, 3]}),
}


def scores(y, pred, prob):
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "roc_auc": roc_auc_score(y, prob), "pr_auc": average_precision_score(y, prob)}


def main():
    df = load_data(ROOT / "data" / "ai_2020.csv")
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    fitted, report = {}, {}
    for name, (model, grid) in MODELS.items():
        gs = GridSearchCV(model, grid, scoring="f1", cv=cv, n_jobs=-1).fit(X_tr, y_tr)
        fitted[name] = gs.best_estimator_
        prob = gs.best_estimator_.predict_proba(X_te)[:, 1]
        report[name] = {**scores(y_te, (prob >= 0.5).astype(int), prob),
                        "cv_f1": gs.best_score_, "params": gs.best_params_}
        print(f"{name:20s} CV F1={gs.best_score_:.3f} | test F1={report[name]['f1']:.3f} "
              f"acc={report[name]['accuracy']:.3f}")

    # Select on cross-validation F1 (never on the test set).
    best = max(report, key=lambda n: report[n]["cv_f1"])
    # Tune the alert threshold on out-of-fold predictions of the training data.
    oof = cross_val_predict(fitted[best], X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y_tr, oof)
    threshold = float(t[(2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-9)).argmax()])
    prob = fitted[best].predict_proba(X_te)[:, 1]
    tuned = scores(y_te, (prob >= threshold).astype(int), prob)
    print(f"\nBest model: {best} | threshold={threshold:.2f} | tuned test F1={tuned['f1']:.3f}")

    rf = fitted["Random Forest"]
    meta = {"best_model": best, "threshold": threshold, "tuned_test": tuned, "models": report,
            "importances": dict(zip(FEATURES, map(float, rf.feature_importances_))),
            "test_size": int(len(y_te)), "test_failures": int(y_te.sum())}
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(fitted[best], ROOT / "models" / "best_model.joblib")
    (ROOT / "models" / "metrics.json").write_text(json.dumps(meta, indent=2, default=float))
    print("Saved models/best_model.joblib and models/metrics.json")


if __name__ == "__main__":
    main()
