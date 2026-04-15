# tuning over 32 configs: RF, XGBoost, LightGBM, GradientBoosting
# evaluate on val only, never touch the test set

import os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import lightgbm as lgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
OUT_DIR      = os.path.dirname(__file__)
os.makedirs(OUT_DIR, exist_ok=True)

# 1. load data (via supervised.py)

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
    df["DestinationPort"] = pd.to_numeric(df["DestinationPort"], errors="coerce").fillna(0).astype(int)

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_val,   y_val   = val_df[FEATURES],   val_df[TARGET]
X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]

# weights for XGBoost/LGBM (equivalent to class_weight=balanced)
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print(f"AI ratio train: {y_train.mean()*100:.1f}% | scale_pos_weight: {scale_pos_weight:.2f}\n")

# 2. evaluation function

results = []

def evaluate_model(name, model, X_tr, y_tr, X_v, y_v, params=None):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    proba_v = model.predict_proba(X_v)[:, 1]
    pred_v  = model.predict(X_v)

    auc  = roc_auc_score(y_v, proba_v)
    ap   = average_precision_score(y_v, proba_v)
    f1   = f1_score(y_v, pred_v, pos_label=1, zero_division=0)
    prec = precision_score(y_v, pred_v, pos_label=1, zero_division=0)
    rec  = recall_score(y_v, pred_v, pos_label=1, zero_division=0)

        # best F1 on threshold curve
    thresholds = np.linspace(0.05, 0.95, 50)
    f1s = [f1_score(y_v, (proba_v>=t).astype(int), pos_label=1, zero_division=0) for t in thresholds]
    best_t  = thresholds[np.argmax(f1s)]
    best_f1 = max(f1s)

    # overfitting gap
    proba_tr  = model.predict_proba(X_tr)[:, 1]
    auc_train = roc_auc_score(y_tr, proba_tr)
    gap       = auc_train - auc

    entry = {
        "name":     name,
        "auc_val":  round(auc, 4),
        "ap_val":   round(ap, 4),
        "f1_val":   round(f1, 4),
        "prec_val": round(prec, 4),
        "rec_val":  round(rec, 4),
        "best_f1":  round(best_f1, 4),
        "best_t":   round(best_t, 2),
        "gap":      round(gap, 4),
        "time_s":   round(elapsed, 1),
        "params":   str(params or ""),
    }
    results.append(entry)
    print(f"  {name:<45} AUC={auc:.4f}  F1={f1:.4f}  BestF1={best_f1:.4f}  Gap={gap:.3f}  ({elapsed:.1f}s)")
    return model, proba_v

# 3. random forest - grid search

print("="*65)
print("A. RANDOM FOREST")
print("="*65)

rf_configs = [
    # baseline v5
    {"n_estimators":200, "max_depth":12, "min_samples_leaf":10, "class_weight":"balanced"},
    # profondeur plus faible
    {"n_estimators":200, "max_depth":8,  "min_samples_leaf":10, "class_weight":"balanced"},
    {"n_estimators":200, "max_depth":6,  "min_samples_leaf":10, "class_weight":"balanced"},
    # profondeur plus haute
    {"n_estimators":200, "max_depth":15, "min_samples_leaf":10, "class_weight":"balanced"},
    {"n_estimators":200, "max_depth":20, "min_samples_leaf":10, "class_weight":"balanced"},
    # min_samples_leaf
    {"n_estimators":200, "max_depth":12, "min_samples_leaf":5,  "class_weight":"balanced"},
    {"n_estimators":200, "max_depth":12, "min_samples_leaf":20, "class_weight":"balanced"},
    {"n_estimators":200, "max_depth":12, "min_samples_leaf":30, "class_weight":"balanced"},
    # n_estimators
    {"n_estimators":100, "max_depth":12, "min_samples_leaf":10, "class_weight":"balanced"},
    {"n_estimators":300, "max_depth":12, "min_samples_leaf":10, "class_weight":"balanced"},
    # class_weight alternatif
    {"n_estimators":200, "max_depth":12, "min_samples_leaf":10, "class_weight":"balanced_subsample"},
    # combinaisons prometteuses
    {"n_estimators":300, "max_depth":10, "min_samples_leaf":15, "class_weight":"balanced"},
    {"n_estimators":300, "max_depth":8,  "min_samples_leaf":20, "class_weight":"balanced"},
    {"n_estimators":200, "max_depth":10, "min_samples_leaf":5,  "class_weight":"balanced"},
]

best_rf_model, best_rf_proba = None, None
for p in rf_configs:
    m = RandomForestClassifier(**p, n_jobs=-1, random_state=42)
    label = f"RF  depth={p['max_depth']} leaf={p['min_samples_leaf']} n={p['n_estimators']} cw={p['class_weight'][:3]}"
    model_out, proba_out = evaluate_model(label, m, X_train, y_train, X_val, y_val, p)
    if best_rf_model is None or results[-1]["auc_val"] > max(r["auc_val"] for r in results[:-1] if r["name"].startswith("RF")):
        best_rf_model, best_rf_proba = model_out, proba_out

# 4. xgboost - grid search

print("\n" + "="*65)
print("B. XGBOOST")
print("="*65)

xgb_configs = [
    {"n_estimators":200, "max_depth":6,  "learning_rate":0.1,  "subsample":0.8, "colsample_bytree":0.8},
    {"n_estimators":200, "max_depth":4,  "learning_rate":0.1,  "subsample":0.8, "colsample_bytree":0.8},
    {"n_estimators":200, "max_depth":8,  "learning_rate":0.1,  "subsample":0.8, "colsample_bytree":0.8},
    {"n_estimators":200, "max_depth":6,  "learning_rate":0.05, "subsample":0.8, "colsample_bytree":0.8},
    {"n_estimators":300, "max_depth":6,  "learning_rate":0.05, "subsample":0.8, "colsample_bytree":0.8},
    {"n_estimators":200, "max_depth":6,  "learning_rate":0.1,  "subsample":0.6, "colsample_bytree":0.6},
    {"n_estimators":200, "max_depth":6,  "learning_rate":0.1,  "subsample":1.0, "colsample_bytree":1.0},
    {"n_estimators":300, "max_depth":4,  "learning_rate":0.05, "subsample":0.8, "colsample_bytree":0.8},
]

best_xgb_model, best_xgb_proba = None, None
for p in xgb_configs:
    m = xgb.XGBClassifier(
        **p,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    label = f"XGB depth={p['max_depth']} lr={p['learning_rate']} n={p['n_estimators']} sub={p['subsample']}"
    model_out, proba_out = evaluate_model(label, m, X_train, y_train, X_val, y_val, p)
    if best_xgb_model is None or results[-1]["auc_val"] > max(r["auc_val"] for r in results if r["name"].startswith("XGB") and r != results[-1]):
        best_xgb_model, best_xgb_proba = model_out, proba_out

# 5. lightgbm - grid search

print("\n" + "="*65)
print("C. LIGHTGBM")
print("="*65)

lgb_configs = [
    {"n_estimators":200, "max_depth":6,  "learning_rate":0.1,  "num_leaves":31,  "subsample":0.8},
    {"n_estimators":200, "max_depth":4,  "learning_rate":0.1,  "num_leaves":15,  "subsample":0.8},
    {"n_estimators":200, "max_depth":8,  "learning_rate":0.1,  "num_leaves":63,  "subsample":0.8},
    {"n_estimators":200, "max_depth":6,  "learning_rate":0.05, "num_leaves":31,  "subsample":0.8},
    {"n_estimators":300, "max_depth":6,  "learning_rate":0.05, "num_leaves":31,  "subsample":0.8},
    {"n_estimators":300, "max_depth":4,  "learning_rate":0.05, "num_leaves":15,  "subsample":0.8},
    {"n_estimators":200, "max_depth":6,  "learning_rate":0.1,  "num_leaves":63,  "subsample":0.6},
]

best_lgb_model, best_lgb_proba = None, None
for p in lgb_configs:
    m = lgb.LGBMClassifier(
        **p,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    label = f"LGB depth={p['max_depth']} lr={p['learning_rate']} leaves={p['num_leaves']} n={p['n_estimators']}"
    model_out, proba_out = evaluate_model(label, m, X_train, y_train, X_val, y_val, p)
    if best_lgb_model is None or results[-1]["auc_val"] > max(r["auc_val"] for r in results if r["name"].startswith("LGB") and r != results[-1]):
        best_lgb_model, best_lgb_proba = model_out, proba_out

# 6. gradient boosting (sklearn)

print("\n" + "="*65)
print("D. GRADIENT BOOSTING")
print("="*65)

gb_configs = [
    {"n_estimators":200, "max_depth":4, "learning_rate":0.1,  "subsample":0.8},
    {"n_estimators":200, "max_depth":3, "learning_rate":0.1,  "subsample":0.8},
    {"n_estimators":300, "max_depth":4, "learning_rate":0.05, "subsample":0.8},
]

for p in gb_configs:
    # GradientBoosting has no class_weight: use sample_weight instead
    sw = np.where(y_train == 1, scale_pos_weight, 1.0)
    m  = GradientBoostingClassifier(**p, random_state=42)
    m.fit(X_train, y_train, sample_weight=sw)
    proba_v = m.predict_proba(X_val)[:, 1]
    pred_v  = m.predict(X_val)
    auc  = roc_auc_score(y_val, proba_v)
    f1   = f1_score(y_val, pred_v, pos_label=1, zero_division=0)
    ap   = average_precision_score(y_val, proba_v)
    prec = precision_score(y_val, pred_v, pos_label=1, zero_division=0)
    rec  = recall_score(y_val, pred_v, pos_label=1, zero_division=0)
    thresholds = np.linspace(0.05, 0.95, 50)
    f1s = [f1_score(y_val, (proba_v>=t).astype(int), pos_label=1, zero_division=0) for t in thresholds]
    best_t  = thresholds[np.argmax(f1s)]
    best_f1 = max(f1s)
    auc_tr  = roc_auc_score(y_train, m.predict_proba(X_train)[:, 1])
    gap     = auc_tr - auc
    label = f"GB  depth={p['max_depth']} lr={p['learning_rate']} n={p['n_estimators']}"
    entry = {"name": label, "auc_val": round(auc,4), "ap_val": round(ap,4),
             "f1_val": round(f1,4), "prec_val": round(prec,4), "rec_val": round(rec,4),
             "best_f1": round(best_f1,4), "best_t": round(best_t,2), "gap": round(gap,4),
             "time_s": 0, "params": str(p)}
    results.append(entry)
    print(f"  {label:<45} AUC={auc:.4f}  F1={f1:.4f}  BestF1={best_f1:.4f}  Gap={gap:.3f}")

# summary table and best model selection

df_results = pd.DataFrame(results)
df_results = df_results.sort_values("auc_val", ascending=False).reset_index(drop=True)

print("\n" + "="*65)
print("SUMMARY TABLE (sorted by val AUC)")
print("="*65)
print(df_results[["name","auc_val","f1_val","best_f1","prec_val","rec_val","gap","time_s"]].to_string(index=True))

best_row   = df_results.iloc[0]
best_name  = best_row["name"]
best_auc   = best_row["auc_val"]
best_f1    = best_row["best_f1"]
print(f"\n>>> MEILLEUR : {best_name}")
print(f"    AUC={best_auc}  BestF1={best_f1}")

# saver le tableau
df_results.to_csv(os.path.join(OUT_DIR, "tuning_results.csv"), index=False)

# evaluate best model on test set

# retrieve best model
print(f"\nFinal evaluation on TEST SET: {best_name}")
params_str = best_row["params"]

# get the right model object
if best_name.startswith("RF"):
    import ast
    p = ast.literal_eval(params_str)
    final_model = RandomForestClassifier(**p, n_jobs=-1, random_state=42)
elif best_name.startswith("XGB"):
    import ast
    p = ast.literal_eval(params_str)
    final_model = xgb.XGBClassifier(**p, scale_pos_weight=scale_pos_weight,
                                    use_label_encoder=False, eval_metric="logloss",
                                    n_jobs=-1, random_state=42, verbosity=0)
elif best_name.startswith("LGB"):
    import ast
    p = ast.literal_eval(params_str)
    final_model = lgb.LGBMClassifier(**p, scale_pos_weight=scale_pos_weight,
                                     n_jobs=-1, random_state=42, verbose=-1)
else:
    import ast
    p = ast.literal_eval(params_str)
    final_model = GradientBoostingClassifier(**p, random_state=42)

# retrain on train+val for final model
X_trainval = pd.concat([X_train, X_val])
y_trainval = pd.concat([y_train, y_val])
final_model.fit(X_trainval, y_trainval)

proba_test  = final_model.predict_proba(X_test)[:, 1]
pred_test   = final_model.predict(X_test)
auc_test    = roc_auc_score(y_test, proba_test)
f1_test     = f1_score(y_test, pred_test, pos_label=1, zero_division=0)
prec_test   = precision_score(y_test, pred_test, pos_label=1, zero_division=0)
rec_test    = recall_score(y_test, pred_test, pos_label=1, zero_division=0)
ap_test     = average_precision_score(y_test, proba_test)

# best threshold on val
thresholds = np.linspace(0.05, 0.95, 50)
# recompute on val with final model (trained on trainval - use original val split)
final_model_val = final_model.__class__(**final_model.get_params())
if hasattr(final_model_val, 'use_label_encoder'):
    pass
final_model_val.fit(X_train, y_train)
proba_val_final = final_model_val.predict_proba(X_val)[:, 1]
f1s_v = [f1_score(y_val, (proba_val_final>=t).astype(int), pos_label=1, zero_division=0) for t in thresholds]
best_t = thresholds[np.argmax(f1s_v)]
pred_test_bt = (proba_test >= best_t).astype(int)
f1_test_bt   = f1_score(y_test, pred_test_bt, pos_label=1, zero_division=0)
prec_test_bt = precision_score(y_test, pred_test_bt, pos_label=1, zero_division=0)
rec_test_bt  = recall_score(y_test, pred_test_bt, pos_label=1, zero_division=0)
cm           = confusion_matrix(y_test, pred_test_bt)

print(f"\n  Test AUC      : {auc_test:.4f}")
print(f"  Test F1 (0.5) : {f1_test:.4f}")
print(f"  Test F1 ({best_t:.2f}): {f1_test_bt:.4f}  Prec={prec_test_bt:.3f}  Rec={rec_test_bt:.3f}")
print(f"  TP={cm[1,1]}  FP={cm[0,1]}  FN={cm[1,0]}  TN={cm[0,0]}")

# session aggregation
WINDOW_SEC = 300
SESSION_THRESHOLD = 0.40
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
SERVICE_MAP = {"https":443,"http":80,"domain":53,"dns":53,"ntp":123,"ssh":22,"ftp":21,
               "smtp":25,"imap":143,"pop3":110,"quic":443,"mdns":5353}
raw_dfs = []
for fpath, label in raw_files.items():
    full_path = os.path.join(PROJECT_ROOT, fpath)
    if os.path.exists(full_path):
        tmp = pd.read_csv(full_path)
        tmp["LabelAI"] = label
        raw_dfs.append(tmp)
raw_all = pd.concat(raw_dfs, ignore_index=True)
raw_all.columns = raw_all.columns.str.strip()
raw_all["Dport"] = pd.to_numeric(raw_all["Dport"].replace(SERVICE_MAP), errors="coerce").fillna(0).astype(int)
raw_all["Dur"]   = pd.to_numeric(raw_all["Dur"], errors="coerce")
raw_all          = raw_all[raw_all["Dur"] > 0].dropna(subset=["Dur","TotPkts","TotBytes"])
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
raw_all["proba_ai"] = final_model.predict_proba(raw_all[FEATURES])[:, 1]
raw_all["pred_ai"]  = (raw_all["proba_ai"] >= best_t).astype(int)
raw_all["session_window"] = raw_all["ts"] // WINDOW_SEC
sessions = raw_all.groupby(["SrcAddr","session_window"]).agg(
    n_flows=("pred_ai","count"), n_ai_flows=("pred_ai","sum"),
    true_label=("LabelAI","max"), mean_proba=("proba_ai","mean"),
).reset_index()
sessions["pct_ai_flows"]    = sessions["n_ai_flows"] / sessions["n_flows"]
sessions["session_pred_ai"] = (sessions["pct_ai_flows"] >= SESSION_THRESHOLD).astype(int)
y_s   = sessions["true_label"]
y_sp  = sessions["session_pred_ai"]
prec_s = precision_score(y_s, y_sp, pos_label=1, zero_division=0)
rec_s  = recall_score(y_s, y_sp,    pos_label=1, zero_division=0)
f1_s   = f1_score(y_s, y_sp,        pos_label=1, zero_division=0)
cm_s   = confusion_matrix(y_s, y_sp)
print(f"\n  Session-level : Prec={prec_s:.3f}  Rec={rec_s:.3f}  F1={f1_s:.3f}  FP={cm_s[0,1]}")

# plots

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f"Tuning - Best model: {best_name[:50]}", fontsize=12, fontweight="bold")

# top 15 models by AUC
ax = axes[0, 0]
top15 = df_results.head(15).copy()
colors = ["gold" if i == 0 else ("steelblue" if n.startswith("RF") else
          "darkorange" if n.startswith("XGB") else
          "green" if n.startswith("LGB") else "purple")
          for i, n in enumerate(top15["name"])]
short_names = [n[:35] for n in top15["name"]]
bars = ax.barh(range(len(top15)), top15["auc_val"], color=colors, edgecolor="white")
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(short_names, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("AUC-ROC (validation)")
ax.set_title("Top 15 models by val AUC")
ax.axvline(top15["auc_val"].min(), color="red", linestyle="--", alpha=0.3)
for bar, val in zip(bars, top15["auc_val"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=7)

# AUC vs gap (overfitting)
ax = axes[0, 1]
families = {"RF": "steelblue", "XGB": "darkorange", "LGB": "green", "GB": "purple"}
for fam, color in families.items():
    sub = df_results[df_results["name"].str.startswith(fam)]
    ax.scatter(sub["gap"], sub["auc_val"], c=color, label=fam, alpha=0.7, s=60)
ax.axvline(0.1, color="red", linestyle="--", alpha=0.5, label="Gap=0.10")
ax.set_xlabel("Overfitting gap (train-val AUC)")
ax.set_ylabel("AUC-ROC validation")
ax.set_title("AUC vs Overfitting (par famille)")
ax.legend()

# AUC vs best F1
ax = axes[0, 2]
for fam, color in families.items():
    sub = df_results[df_results["name"].str.startswith(fam)]
    ax.scatter(sub["auc_val"], sub["best_f1"], c=color, label=fam, alpha=0.7, s=60)
best_point = df_results.iloc[0]
ax.scatter(best_point["auc_val"], best_point["best_f1"], c="gold", s=200, zorder=5,
           edgecolors="black", linewidths=1.5, label="Meilleur")
ax.set_xlabel("AUC-ROC validation")
ax.set_ylabel("Best F1 AI (val)")
ax.set_title("AUC vs BestF1 (par famille)")
ax.legend()

# ROC curve for best model on test
ax = axes[1, 0]
fpr, tpr, _ = roc_curve(y_test, proba_test)
ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC={auc_test:.3f}")
ax.fill_between(fpr, tpr, alpha=0.08, color="steelblue")
ax.plot([0,1],[0,1],"k--",alpha=0.4)
ax.set_title(f"ROC - Best model (test)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

# PR curve for best model
ax = axes[1, 1]
prec_c, rec_c, _ = precision_recall_curve(y_test, proba_test)
ax.plot(rec_c, prec_c, color="darkorange", lw=2, label=f"AP={ap_test:.3f}")
ax.fill_between(rec_c, prec_c, alpha=0.08, color="darkorange")
ax.axhline(y_test.mean(), color="gray", linestyle="--", label=f"Baseline={y_test.mean():.2f}")
ax.set_title("Precision-Recall - Best model (test)")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.legend()

# session confusion matrix
ax = axes[1, 2]
sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Non-AI","AI"], yticklabels=["Non-AI","AI"],
            ax=ax, annot_kws={"size":13})
ax.set_title(f"Session-level Confusion Matrix\nPrec={prec_s:.2f} Rec={rec_s:.2f} F1={f1_s:.2f}")
ax.set_ylabel("Actual")
ax.set_xlabel("Predicted")

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "tuning_results.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")

joblib.dump(final_model, os.path.join(OUT_DIR, "shadow_ai_model_v6.pkl"))

# final summary for the report

print("\n" + "="*65)
print("FINAL SUMMARY - VERSION 6")
print("="*65)
print(f"\n  Best model           : {best_name}")
print(f"  Parameters           : {params_str}")
print(f"\n  AUC-ROC test         : {auc_test:.4f}  (v5 = 0.7491)")
print(f"  F1 AI seuil 0.5      : {f1_test:.4f}  (v5 = 0.5150)")
print(f"  F1 AI seuil {best_t:.2f}     : {f1_test_bt:.4f}  (v5 = 0.5065)")
print(f"  Precision AI         : {prec_test_bt:.4f}  (v5 = 0.372)")
print(f"  Recall AI            : {rec_test_bt:.4f}  (v5 = 0.794)")
print(f"\n  Session-level F1     : {f1_s:.4f}  (v5 = 0.722)")
print(f"  Session Precision    : {prec_s:.4f}  (v5 = 0.601)")
print(f"  Session Recall       : {rec_s:.4f}  (v5 = 0.905)")
print(f"  Faux positifs sess.  : {cm_s[0,1]}   (v5 = 89)")
print(f"\n  Model saved          : Model/Version_6/shadow_ai_model_v6.pkl")
print(f"  Tableau CSV          : Model/Version_6/tuning_results.csv")

# save results to CSV
df_results.to_csv(os.path.join(OUT_DIR, "tuning_results.csv"), index=False)
print(f"\n  Configs tested       : {len(results)}")