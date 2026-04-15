# v6 - switched from Random Forest to LightGBM
# best params found after testing 32 configs in tuning.py:
# depth=6, lr=0.05, num_leaves=31, n_estimators=300
# session F1: 0.751 vs 0.722 in v5, FP: 70 vs 89

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve,
    average_precision_score, precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1. load data (via supervised.py)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
OUT_DIR      = os.path.dirname(__file__)

original_dir = os.getcwd()
os.chdir(PROJECT_ROOT)
namespace = {}
with open(os.path.join(PROJECT_ROOT, "Model/Supervised", "supervised.py")) as f:
    exec(f.read(), namespace)
os.chdir(original_dir)

train_df = namespace["supervised_train"].copy()
val_df   = namespace["supervised_validation"].copy()
test_df  = namespace["supervised_test"].copy()

FEATURES = [
    "FlowDuration", "DestinationPort", "TotalPackets",
    "TotalBytes", "FlowBytesPerSec", "FlowPacketsPerSec", "AveragePacketSize",
]
TARGET = "LabelAI"

# some ports in the CSV are not in SERVICE_MAP, default to 0
for df in [train_df, val_df, test_df]:
    df["DestinationPort"] = pd.to_numeric(
        df["DestinationPort"], errors="coerce"
    ).fillna(0).astype(int)

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val,   y_val   = val_df[FEATURES],   val_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

print("=" * 55)
print("DATASET v6 (via supervised.py)")
print("=" * 55)
full = pd.concat([train_df, val_df, test_df])
print(f"  Total flows  : {len(full)}")
print(f"  AI  (1)      : {(full[TARGET]==1).sum()} ({(full[TARGET]==1).mean()*100:.1f}%)")
print(f"  Non-AI (0)   : {(full[TARGET]==0).sum()} ({(full[TARGET]==0).mean()*100:.1f}%)")
print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")
print(f"  scale_pos_weight : {scale_pos_weight:.2f}")

# 2. training - LightGBM (best params)

model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

print("\nTraining LightGBM (depth=6, lr=0.05, leaves=31, n=300)...")
model.fit(X_train, y_train)
print("Done.")

# flow-level metrics
y_val_proba  = model.predict_proba(X_val)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]
y_train_proba = model.predict_proba(X_train)[:, 1]

auc_val   = roc_auc_score(y_val,   y_val_proba)
auc_test  = roc_auc_score(y_test,  y_test_proba)
auc_train = roc_auc_score(y_train, y_train_proba)
gap       = auc_train - auc_val

print(f"\n  AUC train : {auc_train:.4f}")
print(f"  AUC val   : {auc_val:.4f}")
print(f"  AUC test  : {auc_test:.4f}")
print(f"  Gap       : {gap:.4f}")

# best threshold on val (max F1)
thresholds = np.linspace(0.05, 0.95, 50)
f1s_val    = [f1_score(y_val, (y_val_proba >= t).astype(int),
                       pos_label=1, zero_division=0) for t in thresholds]
best_t     = thresholds[int(np.argmax(f1s_val))]

y_test_bt   = (y_test_proba >= best_t).astype(int)
prec_flow   = precision_score(y_test, y_test_bt, pos_label=1, zero_division=0)
rec_flow    = recall_score(y_test,    y_test_bt, pos_label=1, zero_division=0)
f1_flow     = f1_score(y_test,        y_test_bt, pos_label=1, zero_division=0)
cm_flow     = confusion_matrix(y_test, y_test_bt)

print(f"\n  Best threshold val : {best_t:.2f}")
print(f"  Flow-level  Prec={prec_flow:.3f} | Rec={rec_flow:.3f} | F1={f1_flow:.3f}")
print(f"  TP={cm_flow[1,1]}  FP={cm_flow[0,1]}  FN={cm_flow[1,0]}  TN={cm_flow[0,0]}")

# 3. cross-validation (5-fold, on train+val)

print("\nCross-validation 5-fold...")
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_tv, y_tv, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"  CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 4. session aggregation (5 min, 40% threshold)

SESSION_THRESHOLD = 0.40
WINDOW_SEC        = 300

raw_files = {
    "data/capture_ai_heavy_2.csv":       1,
    "data/capture_normal_ai_2.csv":      1,
    "data/capture_cloud_work_2.csv":     0,
    "data/capture_normal_web_2.csv":     0,
    "data/capture_streaming_2.csv":      0,
    "data/capture_ai_claude.csv":        1,
    "data/capture_ai_gemini_chatty.csv": 1,
    "data/capture_metro.csv":            0,
    "data/capture_youtube.csv":          0,
}

SERVICE_MAP = {
    "https": 443, "http": 80, "domain": 53, "dns": 53,
    "ntp": 123, "ssh": 22, "ftp": 21, "smtp": 25,
    "imap": 143, "pop3": 110, "quic": 443, "mdns": 5353,
}

raw_dfs = []
for fpath, label in raw_files.items():
    full_path = os.path.join(PROJECT_ROOT, fpath)
    if os.path.exists(full_path):
        tmp = pd.read_csv(full_path)
        tmp["LabelAI"] = label
        raw_dfs.append(tmp)

raw_all = pd.concat(raw_dfs, ignore_index=True)
raw_all.columns = raw_all.columns.str.strip()

raw_all["Dport"] = pd.to_numeric(
    raw_all["Dport"].replace(SERVICE_MAP), errors="coerce"
).fillna(0).astype(int)
raw_all["Dur"] = pd.to_numeric(raw_all["Dur"], errors="coerce")
raw_all = raw_all[raw_all["Dur"] > 0].dropna(subset=["Dur", "TotPkts", "TotBytes"])

raw_all["FlowDuration"]      = raw_all["Dur"]
raw_all["DestinationPort"]   = raw_all["Dport"]
raw_all["TotalPackets"]      = raw_all["TotPkts"]
raw_all["TotalBytes"]        = raw_all["TotBytes"]
raw_all["FlowBytesPerSec"]   = raw_all["TotBytes"] / raw_all["Dur"]
raw_all["FlowPacketsPerSec"] = raw_all["TotPkts"]  / raw_all["Dur"]
raw_all["AveragePacketSize"] = raw_all["TotBytes"]  / raw_all["TotPkts"]
raw_all.replace([np.inf, -np.inf], np.nan, inplace=True)
raw_all.dropna(subset=FEATURES, inplace=True)

raw_all["StartTime_parsed"] = pd.to_datetime(
    raw_all["StartTime"], format="%m/%d.%H:%M:%S.%f", errors="coerce"
)
mask = raw_all["StartTime_parsed"].isna()
if mask.any():
    raw_all.loc[mask, "StartTime_parsed"] = pd.to_datetime(
        raw_all.loc[mask, "StartTime"], errors="coerce"
    )
raw_all["ts"] = raw_all["StartTime_parsed"].astype(np.int64) // 10**9

raw_all["proba_ai"] = model.predict_proba(raw_all[FEATURES])[:, 1]
raw_all["pred_ai"]  = (raw_all["proba_ai"] >= best_t).astype(int)
raw_all["session_window"] = raw_all["ts"] // WINDOW_SEC

sessions = raw_all.groupby(["SrcAddr", "session_window"]).agg(
    n_flows       = ("pred_ai", "count"),
    n_ai_flows    = ("pred_ai", "sum"),
    true_label    = ("LabelAI", "max"),
    mean_proba    = ("proba_ai", "mean"),
    total_bytes   = ("TotalBytes", "sum"),
    mean_duration = ("FlowDuration", "mean"),
).reset_index()

sessions["pct_ai_flows"]    = sessions["n_ai_flows"] / sessions["n_flows"]
sessions["session_pred_ai"] = (sessions["pct_ai_flows"] >= SESSION_THRESHOLD).astype(int)

y_sess   = sessions["true_label"]
y_s_pred = sessions["session_pred_ai"]

prec_s = precision_score(y_sess, y_s_pred, pos_label=1, zero_division=0)
rec_s  = recall_score(y_sess,    y_s_pred, pos_label=1, zero_division=0)
f1_s   = f1_score(y_sess,        y_s_pred, pos_label=1, zero_division=0)
cm_s   = confusion_matrix(y_sess, y_s_pred)

print("\n" + "=" * 55)
print(f"SESSION AGGREGATION (window={WINDOW_SEC//60}min, threshold={SESSION_THRESHOLD*100:.0f}%)")
print("=" * 55)
print(f"  Sessions totales : {len(sessions)}")
print(f"  Sessions AI      : {(y_sess==1).sum()}")
print(f"  Sessions Non-AI  : {(y_sess==0).sum()}")
print(f"\n  Precision  : {prec_s:.3f}")
print(f"  Recall     : {rec_s:.3f}")
print(f"  F1         : {f1_s:.3f}")
print(f"  TP={cm_s[1,1]}  FP={cm_s[0,1]}  FN={cm_s[1,0]}  TN={cm_s[0,0]}")

# session threshold sensitivity
print("\n  Session threshold sensitivity:")
print(f"  {'Seuil%':<10} {'Precision':<12} {'Recall':<10} {'F1':<8} {'FP'}")
print("  " + "-" * 50)
for t_sess in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    yp = (sessions["pct_ai_flows"] >= t_sess).astype(int)
    p  = precision_score(y_sess, yp, pos_label=1, zero_division=0)
    r  = recall_score(y_sess,    yp, pos_label=1, zero_division=0)
    f  = f1_score(y_sess,        yp, pos_label=1, zero_division=0)
    fp = confusion_matrix(y_sess, yp)[0, 1]
    marker = " <-- choisi" if t_sess == SESSION_THRESHOLD else ""
    print(f"  {t_sess*100:<10.0f} {p:<12.3f} {r:<10.3f} {f:<8.3f} {fp}{marker}")

# plots

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "Shadow AI Detection v6 - LightGBM + Session Aggregation",
    fontsize=14, fontweight="bold"
)

# 5.1 v5 vs v6 comparison (session-level)
ax = axes[0, 0]
categories = ["Precision", "Recall", "F1"]
v5_vals = [0.601, 0.905, 0.722]   # known v5 results
v6_vals = [prec_s, rec_s, f1_s]
x = np.arange(len(categories))
w = 0.35
b1 = ax.bar(x - w/2, v5_vals, w, label="v5 (RF)",  color="steelblue",   alpha=0.8)
b2 = ax.bar(x + w/2, v6_vals, w, label="v6 (LGB)", color="darkorange", alpha=0.8)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01,
            f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylim(0, 1.1)
ax.set_title("Session-level : v5 (RF) vs v6 (LGB)")
ax.legend()
ax.axhline(0.5, color="red", linestyle="--", alpha=0.3)

# 5.2 flow-level confusion matrix
ax = axes[0, 1]
sns.heatmap(cm_flow, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-AI", "AI"], yticklabels=["Non-AI", "AI"],
            ax=ax, annot_kws={"size": 12})
ax.set_title(f"Confusion Matrix - Flow (seuil={best_t:.2f})\nFP={cm_flow[0,1]}")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")

# 5.3 session-level confusion matrix
ax = axes[0, 2]
sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Non-AI", "AI"], yticklabels=["Non-AI", "AI"],
            ax=ax, annot_kws={"size": 12})
ax.set_title(f"Confusion Matrix - Session (seuil={SESSION_THRESHOLD*100:.0f}%)\nFP={cm_s[0,1]}")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")

# 5.4 session threshold vs metrics
ax = axes[1, 0]
sess_thresholds = np.linspace(0.05, 0.95, 50)
precs_s, recs_s, f1s_s = [], [], []
for t in sess_thresholds:
    yp = (sessions["pct_ai_flows"] >= t).astype(int)
    precs_s.append(precision_score(y_sess, yp, pos_label=1, zero_division=0))
    recs_s.append(recall_score(y_sess,     yp, pos_label=1, zero_division=0))
    f1s_s.append(f1_score(y_sess,          yp, pos_label=1, zero_division=0))
ax.plot(sess_thresholds, precs_s, label="Precision", color="green")
ax.plot(sess_thresholds, recs_s,  label="Recall",    color="red")
ax.plot(sess_thresholds, f1s_s,   label="F1",        color="steelblue", lw=2)
ax.axvline(SESSION_THRESHOLD, color="black", linestyle="--", alpha=0.6,
           label=f"Seuil choisi = {SESSION_THRESHOLD}")
ax.set_title("Session threshold vs metrics")
ax.set_xlabel("% flows AI dans la session")
ax.set_ylabel("Score")
ax.legend(fontsize=8)

# 5.5 % AI flows per session
ax = axes[1, 1]
ai_sessions    = sessions[sessions["true_label"] == 1]["pct_ai_flows"]
nonai_sessions = sessions[sessions["true_label"] == 0]["pct_ai_flows"]
ax.hist(nonai_sessions, bins=20, alpha=0.6, color="steelblue",  label="Non-AI", density=True)
ax.hist(ai_sessions,    bins=20, alpha=0.6, color="darkorange", label="AI",     density=True)
ax.axvline(SESSION_THRESHOLD, color="black", linestyle="--",
           label=f"Seuil {SESSION_THRESHOLD*100:.0f}%")
ax.set_title("Distribution % flows AI par session")
ax.set_xlabel("% flows classified AI")
ax.set_ylabel("Density")
ax.legend(fontsize=8)

# 5.6 ROC curve + feature importance
ax = axes[1, 2]
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"LGB AUC={auc_test:.3f}")
ax.fill_between(fpr, tpr, alpha=0.07, color="darkorange")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_title("Courbe ROC (flow-level, test)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "shadow_ai_v6_results.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")

# feature importance

print("\n" + "=" * 55)
print("FEATURE IMPORTANCE (LightGBM gain)")
print("=" * 55)
importances = model.feature_importances_
for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    bar = "#" * int(imp / max(importances) * 30)
    print(f"  {feat:<25} {imp:>6.0f}  {bar}")

# summary final

print("\n" + "=" * 55)
print("RESUME v6")
print("=" * 55)
print(f"  Model        : LightGBM depth=6, lr=0.05, leaves=31, n=300")
print(f"  CV AUC       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  AUC test     : {auc_test:.4f}")
print(f"  Overfitting  : {gap:.4f} (train-val gap)")
print(f"  Flow  seuil {best_t:.2f} : Prec={prec_flow:.3f} | Rec={rec_flow:.3f} | F1={f1_flow:.3f} | FP={cm_flow[0,1]}")
print(f"  Session 40%  : Prec={prec_s:.3f} | Rec={rec_s:.3f} | F1={f1_s:.3f} | FP={cm_s[0,1]}")

# save model
out_model = os.path.join(OUT_DIR, "shadow_ai_model_v6.pkl")
joblib.dump(model, out_model)
print(f"\nModel saved: Model/Version_6/shadow_ai_model_v6.pkl")