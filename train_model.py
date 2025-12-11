from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_curve,
)

from xgboost import XGBClassifier
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PROC_DIR = ROOT / "data" / "processed"
TRAIN_PATH = PROC_DIR / "training_data.csv"
MODEL_XGB_PATH = PROC_DIR / "model_is_real_xgb.pkl"
PLOTS_DIR = ROOT / "plots"

FEATURE_COLS = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_model_snr",
    "koi_teq",
    "teff",
    "radius",
    "mass",
    "dens",
    "log_koi_prad",
    "log_koi_period",
    "log_koi_teq",
    "dur_over_period",
    "depth_over_radius2",
    "duration_over_radius",
]


def load_training_data() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH)


def get_feature_matrix(df: pd.DataFrame):
    df_a = df[df["is_real_planet"].notna()].copy()
    X = df_a[[c for c in FEATURE_COLS if c in df_a.columns]]
    y = df_a["is_real_planet"].astype(int)
    mask = X.notna().all(axis=1)
    return X[mask], y[mask], df_a[mask]


def train_xgb_model(X: pd.DataFrame, y: pd.Series, df_ml: pd.DataFrame):
    X_tmp, X_test, y_tmp, y_test, df_tmp, df_test = train_test_split(
        X,
        y,
        df_ml,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
        X_tmp,
        y_tmp,
        df_tmp,
        test_size=0.25,
        random_state=42,
        stratify=y_tmp,
    )

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.7,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0.5,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
    )

    print("Training model…")
    xgb.fit(X_train, y_train)

    val_proba = xgb.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.2, 0.8, 13)

    best_t_f1 = 0.5
    best_f1 = -1.0
    best_t_high_recall = None
    best_f1_high_recall = -1.0

    print("\nThreshold sweep (validation, class=REAL PLANET):")
    for t in thresholds:
        pred_val = (val_proba >= t).astype(int)
        f1 = f1_score(y_val, pred_val, pos_label=1)
        prec = precision_score(y_val, pred_val, pos_label=1, zero_division=0)
        rec = recall_score(y_val, pred_val, pos_label=1, zero_division=0)
        print(f"  t={t:.2f}  F1={f1:.3f}  P={prec:.3f}  R={rec:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_t_f1 = t
        if rec >= 0.93 and f1 > best_f1_high_recall:
            best_f1_high_recall = f1
            best_t_high_recall = t

    if best_t_high_recall is not None:
        chosen_raw = best_t_high_recall
    else:
        chosen_raw = best_t_f1

    adjusted_t = max(0.0, min(1.0, chosen_raw - 0.02))
    print(f"\nChosen decision threshold (after -0.02 adjustment): {adjusted_t:.2f}\n")

    test_proba = xgb.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= adjusted_t).astype(int)

    acc = accuracy_score(y_test, test_pred)
    f1_pos = f1_score(y_test, test_pred, pos_label=1)
    roc = roc_auc_score(y_test, test_proba)
    pr = average_precision_score(y_test, test_proba)

    print("Test set summary")
    print("----------------")
    print(f"Accuracy:  {acc:.3f}")
    print(f"F1 (pos):  {f1_pos:.3f}")
    print(f"ROC AUC:   {roc:.3f}")
    print(f"PR  AUC:   {pr:.3f}\n")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, test_pred))
    print("\nClassification report:")
    print(classification_report(y_test, test_pred))

    xgb.fit(X, y)
    bundle = {
        "model": xgb,
        "threshold": float(adjusted_t),
        "features": list(X.columns),
    }
    joblib.dump(bundle, MODEL_XGB_PATH)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Feature importance
    importances = xgb.feature_importances_
    feat_imp = sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True)
    names = [n for n, _ in feat_imp]
    vals = [v for _, v in feat_imp]

    plt.figure(figsize=(8, 5))
    plt.bar(range(len(names)), vals)
    plt.xticks(range(len(names)), names, rotation=60, ha="right")
    plt.ylabel("Importance")
    plt.title("Feature importance (XGBoost)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importances.png", dpi=200)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curve.png", dpi=200)
    plt.close()

    # Probability histogram by true class
    plt.figure(figsize=(8, 5))
    plt.hist(
        test_proba[y_test == 0],
        bins=30,
        alpha=0.7,
        label="True FALSE POSITIVE",
    )
    plt.hist(
        test_proba[y_test == 1],
        bins=30,
        alpha=0.7,
        label="True REAL PLANET",
    )
    plt.xlabel("Predicted probability of REAL PLANET")
    plt.ylabel("Count")
    plt.title("Predicted probabilities by true class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "proba_hist_by_true_class.png", dpi=200)
    plt.close()

    # Attach metadata for per-type plots
    df_test = df_test.copy()
    df_test["y_true"] = y_test.values
    df_test["y_pred"] = test_pred
    df_test["proba"] = test_proba
    df_test["size_class"] = df_test["size_class"].astype(str)
    df_test["temp_regime"] = df_test["temp_regime"].astype(str)
    df_test["combined_type_label"] = df_test["combined_type_label"].fillna(
        "Unknown world"
    )

    # Probability vs temperature regime
    temp_mask = df_test["temp_regime"].notna()
    df_temp = df_test[temp_mask]
    regimes = sorted(df_temp["temp_regime"].unique())
    regime_to_x = {r: i for i, r in enumerate(regimes)}

    plt.figure(figsize=(8, 5))
    for r in regimes:
        sub = df_temp[df_temp["temp_regime"] == r]
        x_vals = np.full(len(sub), regime_to_x[r], dtype=float)
        x_vals += np.random.uniform(-0.15, 0.15, size=len(sub))
        plt.scatter(x_vals, sub["proba"], s=12, label=r)
    plt.axhline(adjusted_t, linestyle="--")
    plt.xticks(range(len(regimes)), regimes)
    plt.xlabel("Temperature regime (F/W/G/R)")
    plt.ylabel("Predicted probability of REAL PLANET")
    plt.title("Probability by temperature regime")
    plt.legend(title="Regime", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scatter_regime_vs_proba.png", dpi=200)
    plt.close()

    # Probability vs size class
    size_mask = df_test["size_class"].notna()
    df_size = df_test[size_mask]
    size_classes = sorted(df_size["size_class"].unique())
    size_to_x = {s: i for i, s in enumerate(size_classes)}

    plt.figure(figsize=(9, 5))
    for s in size_classes:
        sub = df_size[df_size["size_class"] == s]
        x_vals = np.full(len(sub), size_to_x[s], dtype=float)
        x_vals += np.random.uniform(-0.15, 0.15, size=len(sub))
        plt.scatter(x_vals, sub["proba"], s=12, label=s)
    plt.axhline(adjusted_t, linestyle="--")
    plt.xticks(range(len(size_classes)), size_classes, rotation=45, ha="right")
    plt.xlabel("Planet size class")
    plt.ylabel("Predicted probability of REAL PLANET")
    plt.title("Probability by size class")
    plt.legend(title="Size", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scatter_size_vs_proba.png", dpi=200)
    plt.close()

#performance by planet type
    recalls_real = []
    recalls_fp = []
    types = []

    for tlabel, group in df_test.groupby("combined_type_label"):
        real_mask = group["y_true"] == 1
        fp_mask = group["y_true"] == 0

        if real_mask.sum() > 0:
            r_real = (group.loc[real_mask, "y_pred"] == 1).mean()
        else:
            r_real = np.nan

        if fp_mask.sum() > 0:
            r_fp = (group.loc[fp_mask, "y_pred"] == 0).mean()
        else:
            r_fp = np.nan

        types.append(tlabel)
        recalls_real.append(r_real)
        recalls_fp.append(r_fp)

    x = np.arange(len(types))
    width = 0.4

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, recalls_real, width, label="Recall on REAL planets")
    plt.bar(x + width / 2, recalls_fp, width, label="Correct on FALSE POSITIVES")
    plt.xticks(x, types, rotation=60, ha="right")
    plt.ylabel("Accuracy / recall")
    plt.title("Performance by planet type (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "performance_by_planet_type.png", dpi=200)
    plt.close()


def main():
    df = load_training_data()
    X, y, df_ml = get_feature_matrix(df)
    train_xgb_model(X, y, df_ml)


if __name__ == "__main__":
    main()
