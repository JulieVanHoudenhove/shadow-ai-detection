# v4 - new captures added (claude, gemini_chatty, metro, youtube)
# preprocessing is now in supervised.py to avoid repeating it everywhere

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score,
    average_precision_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# 1. load dataset via supervised.py (without modifying it)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
OUT_DIR      = os.path.dirname(__file__)

# cd to project root so relative paths in supervised.py resolve correctly
original_dir = os.getcwd()
os.chdir(PROJECT_ROOT)

supervised_script = os.path.join(PROJECT_ROOT, "Model/Supervised", "supervised.py")
namespace = {}
with open(supervised_script, "r") as f:
    exec(f.read(), namespace)

os.chdir(original_dir)

# get the splits produced by supervised.py
train_df = namespace["supervised_train"]
val_df   = namespace["supervised_validation"]
test_df  = namespace["supervised_test"]

FEATURES = [
    "FlowDuration", "DestinationPort", "TotalPackets",
    "TotalBytes", "FlowBytesPerSec", "FlowPacketsPerSec", "AveragePacketSize",
]
TARGET = "LabelAI"

# force-convert DestinationPort (some service names not in supervised.py stay as strings, e.g. imaps, ftps)
for df_split in [train_df, val_df, test_df]:
    df_split["DestinationPort"] = pd.to_numeric(
        df_split["DestinationPort"], errors="coerce"
    ).fillna(0).astype(int)

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val,   y_val   = val_df[FEATURES],   val_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

print("="*55)
print("DATASET v4 (via supervised.py)")
print("="*55)
full = pd.concat([train_df, val_df, test_df])
print(f"  Total flows  : {len(full)}")
print(f"  AI  (1)      : {(full[TARGET]==1).sum()} ({(full[TARGET]==1).mean()*100:.1f}%)")
print(f"  Non-AI (0)   : {(full[TARGET]==0).sum()} ({(full[TARGET]==0).mean()*100:.1f}%)")
print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

# 2. baseline: DummyClassifier

dummy = DummyClassifier(strategy="most_frequent", random_state=42)
dummy.fit(X_train, y_train)
dummy_pred = dummy.predict(X_test)
dummy_f1   = f1_score(y_test, dummy_pred, pos_label=1, zero_division=0)
print(f"\n  Baseline DummyClassifier F1(AI) : {dummy_f1:.4f}")

# random forest

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    max_depth=12,
    min_samples_leaf=10,
    n_jobs=-1,
    random_state=42,
)

print("\nTraining Random Forest v4...")
model.fit(X_train, y_train)
print("Done.")

# 4. val + test evaluation

def evaluate(m, X, y, label=""):
    y_pred  = m.predict(X)
    y_proba = m.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_proba)
    ap  = average_precision_score(y, y_proba)
    f1  = f1_score(y, y_pred, pos_label=1)
    print(f"\n--- {label} ---")
    print(classification_report(y, y_pred, target_names=["Non-AI", "AI"]))
    print(f"  AUC-ROC : {auc:.4f}  |  Avg Precision : {ap:.4f}  |  F1(AI) : {f1:.4f}")
    return y_pred, y_proba, auc, ap, f1

print("\n" + "="*55)
print("RESULTATS v4")
print("="*55)
y_val_pred,  y_val_proba,  auc_val,  ap_val,  f1_val  = evaluate(model, X_val,  y_val,  "VALIDATION")
y_test_pred, y_test_proba, auc_test, ap_test, f1_test = evaluate(model, X_test, y_test, "TEST")

train_proba = model.predict_proba(X_train)[:, 1]
auc_train   = roc_auc_score(y_train, train_proba)
gap         = auc_train - auc_val
print(f"\n  Train AUC : {auc_train:.4f}  |  Val AUC : {auc_val:.4f}  |  Gap : {gap:.4f}  {'OVERFIT' if gap > 0.1 else 'OK'}")

# best threshold on val
thresholds = np.linspace(0.05, 0.95, 50)
f1s_val = [f1_score(y_val, (y_val_proba>=t).astype(int), pos_label=1, zero_division=0) for t in thresholds]
best_t  = thresholds[np.argmax(f1s_val)]
y_test_best    = (y_test_proba >= best_t).astype(int)
f1_best_test   = f1_score(y_test, y_test_best, pos_label=1, zero_division=0)
print(f"\n  Meilleur seuil (val) : {best_t:.2f}  ->  F1 test : {f1_best_test:.4f}")

# 5. cross-validation 5-FOLD

print("\nCross-validation 5-fold...")
X_full = pd.concat([X_train, X_val, X_test])
y_full = pd.concat([y_train, y_val, y_test])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
from sklearn.metrics import make_scorer
cv_results = cross_validate(
    model, X_full, y_full, cv=cv,
    scoring={
        "roc_auc":   "roc_auc",
        "f1":        make_scorer(f1_score, pos_label=1, zero_division=0),
        "precision": make_scorer(precision_score, pos_label=1, zero_division=0),
        "recall":    make_scorer(recall_score, pos_label=1, zero_division=0),
    },
    return_train_score=True,
    n_jobs=-1,
    error_score=np.nan,
)

cv_auc = np.nanmean(cv_results["test_roc_auc"])
cv_f1  = np.nanmean(cv_results["test_f1"])
cv_gap = np.nanmean(cv_results["train_roc_auc"]) - cv_auc
print(f"  CV AUC : {cv_auc:.4f} +/- {np.nanstd(cv_results['test_roc_auc']):.4f}")
print(f"  CV F1  : {cv_f1:.4f} +/- {np.nanstd(cv_results['test_f1']):.4f}")
print(f"  CV Gap : {cv_gap:.4f}  {'OVERFIT' if cv_gap > 0.1 else 'OK'}")

# plots

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Shadow AI Detection v4 - New captures (Claude + Gemini)", fontsize=14, fontweight="bold")

# 6.1 class distribution
ax = axes[0, 0]
class_counts = full[TARGET].value_counts().sort_index()
bars = ax.bar(["Non-AI (0)", "AI (1)"], class_counts.values,
              color=["steelblue", "darkorange"], edgecolor="white")
for bar, val in zip(bars, class_counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(val), ha="center", va="bottom", fontsize=11)
ax.set_title("Distribution des classes")
ax.set_ylabel("Nombre de flows")

# 6.2 ROC curve (val vs test)
ax = axes[0, 1]
fpr_v, tpr_v, _ = roc_curve(y_val,  y_val_proba)
fpr_t, tpr_t, _ = roc_curve(y_test, y_test_proba)
ax.plot(fpr_v, tpr_v, color="darkorange", lw=2, label=f"Val  AUC={auc_val:.3f}")
ax.plot(fpr_t, tpr_t, color="steelblue",  lw=2, label=f"Test AUC={auc_test:.3f}")
ax.fill_between(fpr_t, tpr_t, alpha=0.07, color="steelblue")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("Courbe ROC")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

# 6.3 precision-recall
ax = axes[0, 2]
prec, rec, _ = precision_recall_curve(y_test, y_test_proba)
ax.plot(rec, prec, color="steelblue", lw=2, label=f"AP={ap_test:.3f}")
ax.fill_between(rec, prec, alpha=0.07, color="steelblue")
ax.axhline(y_test.mean(), color="gray", linestyle="--", label=f"Baseline={y_test.mean():.2f}")
ax.set_title("Precision-Recall (test)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()

# 6.4 confusion matrix (best threshold)
ax = axes[1, 0]
cm = confusion_matrix(y_test, y_test_best)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-AI", "AI"],
            yticklabels=["Non-AI", "AI"], ax=ax, annot_kws={"size": 13})
ax.set_title(f"Confusion Matrix (seuil={best_t:.2f})")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")

# 6.5 threshold vs metrics (on val)
ax = axes[1, 1]
precs_v = [precision_score(y_val, (y_val_proba>=t).astype(int), pos_label=1, zero_division=0) for t in thresholds]
recs_v  = [recall_score(y_val,   (y_val_proba>=t).astype(int), pos_label=1, zero_division=0)  for t in thresholds]
ax.plot(thresholds, precs_v, label="Precision", color="green")
ax.plot(thresholds, recs_v,  label="Recall",    color="red")
ax.plot(thresholds, f1s_val, label="F1",        color="steelblue", lw=2)
ax.axvline(best_t, color="black", linestyle="--", alpha=0.6, label=f"Best = {best_t:.2f}")
ax.set_title("Threshold vs metrics (validation)")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Score")
ax.legend(fontsize=8)

# 6.6 cross-val per fold
ax = axes[1, 2]
cv_df = pd.DataFrame({
    "AUC-ROC":   cv_results["test_roc_auc"],
    "F1":        cv_results["test_f1"],
    "Precision": cv_results["test_precision"],
    "Recall":    cv_results["test_recall"],
})
cv_df.index = [f"Fold {i+1}" for i in range(5)]
cv_df.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
ax.set_title("Cross-validation 5-fold")
ax.set_ylim(0, 1)
ax.set_xticklabels(cv_df.index, rotation=0)
ax.legend(loc="lower right", fontsize=8)
ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "shadow_ai_v4_results.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")

# summary comparatif

print("\n" + "="*55)
print("COMPARAISON v3 -> v4")
print("="*55)
print(f"  Nouvelles captures : claude (+62), gemini_chatty (+1722), metro (+2614), youtube (+62)")
print(f"  v3 : ~15k flows | v4 : {len(full)} flows")
print(f"\n  AUC-ROC test  : v3=0.7202  ->  v4={auc_test:.4f}")
print(f"  F1 AI (0.5)   : v3=0.4391  ->  v4={f1_test:.4f}")
print(f"  F1 (best seuil): v3=0.4446 ->  v4={f1_best_test:.4f}")
print(f"  Overfitting   : v3=0.106   ->  v4={gap:.4f}")
print(f"  CV AUC        : v3=0.735   ->  v4={cv_auc:.4f}")

joblib.dump(model, os.path.join(OUT_DIR, "shadow_ai_model_v4.pkl"))
print(f"\nModel saved: Model/Version_4/shadow_ai_model_v4.pkl")