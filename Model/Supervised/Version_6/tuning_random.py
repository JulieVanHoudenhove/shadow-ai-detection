# randomized search on LightGBM, 30 combos x 5-fold
# check whether v6 params are actually optimal

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
OUT_DIR      = os.path.dirname(__file__)

# load data

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

for df in [train_df, val_df, test_df]:
    df["DestinationPort"] = pd.to_numeric(
        df["DestinationPort"], errors="coerce"
    ).fillna(0).astype(int)

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val,   y_val   = val_df[FEATURES],   val_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

# train+val for internal CV in the search
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])

print("=" * 60)
print("RANDOMIZED SEARCH - LightGBM (30 combinaisons x 5-fold)")
print("=" * 60)
print(f"  Train+Val : {len(X_tv)} flows | Test : {len(X_test)} flows")

# randomized search

param_grid = {
    "n_estimators":       [200, 300, 500],
    "max_depth":          [6, 8, 10, -1],
    "learning_rate":      [0.05, 0.1, 0.2],
    "num_leaves":         [31, 63, 127],
    "min_child_samples":  [10, 20, 50],
    "scale_pos_weight":   [2, 3, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- F1 search ---
print("\nOptimisation F1 (30 iter)...")
search_f1 = RandomizedSearchCV(
    lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
    param_distributions=param_grid,
    n_iter=30,
    scoring="f1",
    cv=cv,
    n_jobs=2,
    random_state=42,
    verbose=1,
)
search_f1.fit(X_tv, y_tv)

# --- AUC search (same 30 combos, different scoring) ---
print("Optimisation AUC (30 iter)...")
search_auc = RandomizedSearchCV(
    lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=1),
    param_distributions=param_grid,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    n_jobs=2,
    random_state=99,
    verbose=1,
)
search_auc.fit(X_tv, y_tv)

print(f"\n  Meilleur F1  CV : {search_f1.best_score_:.4f}")
print(f"  Params F1       : {search_f1.best_params_}")
print(f"\n  Meilleur AUC CV : {search_auc.best_score_:.4f}")
print(f"  Params AUC      : {search_auc.best_params_}")

# 3. evaluate both candidates on test set

def eval_candidate(name, params, X_tr, y_tr, X_v, y_v, X_te, y_te):
    """Entraîne sur train, seuil sur val, évalue sur test."""
    m = lgb.LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr)

    proba_v  = m.predict_proba(X_v)[:, 1]
    proba_te = m.predict_proba(X_te)[:, 1]
    proba_tr = m.predict_proba(X_tr)[:, 1]

    auc_val   = roc_auc_score(y_v,  proba_v)
    auc_test  = roc_auc_score(y_te, proba_te)
    auc_train = roc_auc_score(y_tr, proba_tr)
    gap       = auc_train - auc_val

    # best threshold on val
    thresholds = np.linspace(0.05, 0.95, 50)
    f1s = [f1_score(y_v, (proba_v >= t).astype(int), pos_label=1, zero_division=0)
           for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]

    pred_te = (proba_te >= best_t).astype(int)
    prec    = precision_score(y_te, pred_te, pos_label=1, zero_division=0)
    rec     = recall_score(y_te,    pred_te, pos_label=1, zero_division=0)
    f1_te   = f1_score(y_te,        pred_te, pos_label=1, zero_division=0)
    cm      = confusion_matrix(y_te, pred_te)

    print(f"\n  [{name}]")
    print(f"    AUC val={auc_val:.4f} | AUC test={auc_test:.4f} | gap={gap:.4f}")
    print(f"    Seuil={best_t:.2f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1_te:.3f}")
    print(f"    TP={cm[1,1]}  FP={cm[0,1]}  FN={cm[1,0]}  TN={cm[0,0]}")

    return m, best_t, auc_val, auc_test, f1_te, gap, cm

print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

# current v6 reference
v6_params = {
    "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
    "num_leaves": 31, "subsample": 0.8,
    "scale_pos_weight": round((y_train == 0).sum() / (y_train == 1).sum(), 2),
}
m_v6, t_v6, auc_v_v6, auc_t_v6, f1_v6, gap_v6, cm_v6 = eval_candidate(
    "v6 baseline", v6_params, X_train, y_train, X_val, y_val, X_test, y_test
)

m_f1, t_f1, auc_v_f1, auc_t_f1, f1_f1, gap_f1, cm_f1 = eval_candidate(
    "Best F1 (RandomSearch)", search_f1.best_params_, X_train, y_train, X_val, y_val, X_test, y_test
)

m_auc, t_auc, auc_v_auc, auc_t_auc, f1_auc, gap_auc, cm_auc = eval_candidate(
    "Best AUC (RandomSearch)", search_auc.best_params_, X_train, y_train, X_val, y_val, X_test, y_test
)

# pick the best candidate

candidates = [
    ("v6 actuel",    m_v6,  t_v6,  auc_v_v6,  auc_t_v6,  f1_v6,  gap_v6,  cm_v6),
    ("Best F1",      m_f1,  t_f1,  auc_v_f1,  auc_t_f1,  f1_f1,  gap_f1,  cm_f1),
    ("Best AUC",     m_auc, t_auc, auc_v_auc, auc_t_auc, f1_auc, gap_auc, cm_auc),
]

# composite score: 60% AUC + 40% F1 (penalizes overfitting if gap > 0.15)
def composite(auc_v, f1_te, gap):
    penalty = max(0, gap - 0.15) * 0.5
    return 0.6 * auc_v + 0.4 * f1_te - penalty

scores = [(name, composite(auc_v, f1_te, gap), m, t, auc_v, auc_t, f1_te, gap, cm)
          for name, m, t, auc_v, auc_t, f1_te, gap, cm in candidates]
scores.sort(key=lambda x: -x[1])

best_name, best_score, best_model, best_t, best_auc_v, best_auc_t, best_f1, best_gap, best_cm = scores[0]

print("\n" + "=" * 60)
print("CLASSEMENT (score composite = 60% AUC val + 40% F1 test)")
print("=" * 60)
for rank, (name, score, *_) in enumerate(scores, 1):
    print(f"  {rank}. {name:<25} score={score:.4f}")

print(f"\n  Gagnant : {best_name}")

# session aggregation for best model

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
    fp = os.path.join(PROJECT_ROOT, fpath)
    if os.path.exists(fp):
        tmp = pd.read_csv(fp)
        tmp["LabelAI"] = label
        raw_dfs.append(tmp)

raw_all = pd.concat(raw_dfs, ignore_index=True)
raw_all.columns = raw_all.columns.str.strip()
raw_all["Dport"] = pd.to_numeric(raw_all["Dport"].replace(SERVICE_MAP), errors="coerce").fillna(0).astype(int)
raw_all["Dur"]   = pd.to_numeric(raw_all["Dur"], errors="coerce")
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
raw_all["StartTime_parsed"] = pd.to_datetime(raw_all["StartTime"], format="%m/%d.%H:%M:%S.%f", errors="coerce")
mask = raw_all["StartTime_parsed"].isna()
if mask.any():
    raw_all.loc[mask, "StartTime_parsed"] = pd.to_datetime(raw_all.loc[mask, "StartTime"], errors="coerce")
raw_all["ts"] = raw_all["StartTime_parsed"].astype(np.int64) // 10**9

raw_all["proba_ai"] = best_model.predict_proba(raw_all[FEATURES])[:, 1]
raw_all["pred_ai"]  = (raw_all["proba_ai"] >= best_t).astype(int)
raw_all["session_window"] = raw_all["ts"] // WINDOW_SEC

sessions = raw_all.groupby(["SrcAddr", "session_window"]).agg(
    n_flows    = ("pred_ai", "count"),
    n_ai_flows = ("pred_ai", "sum"),
    true_label = ("LabelAI", "max"),
    mean_proba = ("proba_ai", "mean"),
).reset_index()
sessions["pct_ai_flows"]    = sessions["n_ai_flows"] / sessions["n_flows"]
sessions["session_pred_ai"] = (sessions["pct_ai_flows"] >= SESSION_THRESHOLD).astype(int)

y_sess   = sessions["true_label"]
y_s_pred = sessions["session_pred_ai"]

prec_s = precision_score(y_sess, y_s_pred, pos_label=1, zero_division=0)
rec_s  = recall_score(y_sess,    y_s_pred, pos_label=1, zero_division=0)
f1_s   = f1_score(y_sess,        y_s_pred, pos_label=1, zero_division=0)
cm_s   = confusion_matrix(y_sess, y_s_pred)

print("\n" + "=" * 60)
print(f"SESSION-LEVEL ({best_name})")
print("=" * 60)
print(f"  Precision : {prec_s:.3f}  |  Recall : {rec_s:.3f}  |  F1 : {f1_s:.3f}")
print(f"  TP={cm_s[1,1]}  FP={cm_s[0,1]}  FN={cm_s[1,0]}  TN={cm_s[0,0]}")

# comparison plots

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("RandomizedSearch vs current v6 - Test results", fontsize=13, fontweight="bold")

labels_plot   = ["v6 actuel", "Best F1", "Best AUC"]
auc_vals_plot = [auc_t_v6, auc_t_f1, auc_t_auc]
f1_vals_plot  = [f1_v6,    f1_f1,    f1_auc]
gap_vals_plot = [gap_v6,   gap_f1,   gap_auc]
colors        = ["steelblue", "darkorange", "green"]

# AUC
ax = axes[0]
bars = ax.bar(labels_plot, auc_vals_plot, color=colors, alpha=0.8)
for b, v in zip(bars, auc_vals_plot):
    ax.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
ax.set_ylim(0.68, 0.78)
ax.set_title("AUC-ROC (test)")
ax.axhline(auc_t_v6, color="steelblue", linestyle="--", alpha=0.4)

# F1
ax = axes[1]
bars = ax.bar(labels_plot, f1_vals_plot, color=colors, alpha=0.8)
for b, v in zip(bars, f1_vals_plot):
    ax.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
ax.set_ylim(0.45, 0.60)
ax.set_title("F1 AI (seuil optimal, test)")
ax.axhline(f1_v6, color="steelblue", linestyle="--", alpha=0.4)

# gap (overfitting)
ax = axes[2]
bars = ax.bar(labels_plot, gap_vals_plot, color=colors, alpha=0.8)
for b, v in zip(bars, gap_vals_plot):
    ax.text(b.get_x() + b.get_width()/2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
ax.set_title("Overfitting gap (train-val AUC)")
ax.axhline(0.10, color="red", linestyle="--", alpha=0.5, label="objectif < 0.10")
ax.legend(fontsize=8)

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "tuning_random_results.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")

# final summary

print("\n" + "=" * 60)
print("FINAL SUMMARY - RandomizedSearch")
print("=" * 60)
print(f"\n  v6 baseline    : AUC={auc_t_v6:.4f} | F1={f1_v6:.4f} | Gap={gap_v6:.4f}")
print(f"  Best F1 search : AUC={auc_t_f1:.4f} | F1={f1_f1:.4f} | Gap={gap_f1:.4f}")
print(f"  Best AUC search: AUC={auc_t_auc:.4f} | F1={f1_auc:.4f} | Gap={gap_auc:.4f}")
print(f"\n  Gagnant  : {best_name}")
print(f"  Session  : Prec={prec_s:.3f} | Rec={rec_s:.3f} | F1={f1_s:.3f} | FP={cm_s[0,1]}")

if best_name != "v6 actuel":
    print(f"\n  Best params found -> copy into Train_v6.py:")
    print(f"  {scores[0][2].get_params()}")
    joblib.dump(best_model, os.path.join(OUT_DIR, "shadow_ai_model_v6_random.pkl"))
    print(f"  Model saved: shadow_ai_model_v6_random.pkl")
else:
    print(f"\n  Current v6 config remains optimal. No changes needed.")