# v7 - added SourcePort + encoding of categorical columns (Proto, Dir, State)
# supervised.py was dropping them so we load the CSV files directly
# SrcAddr and DstAddr excluded: too specific to captures, guaranteed overfitting
# StartTime also excluded: no stable temporal pattern across captures from different dates

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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_curve,
    average_precision_score,
)
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
OUT_DIR      = os.path.dirname(__file__)

# 1. load raw CSV files (same files as supervised.py)

RAW_FILES = {
    "data/capture_ai_heavy_2.csv":       1,
    "data/capture_normal_ai_2.csv":      1,
    "data/capture_cloud_work_2.csv":     0,
    "data/capture_normal_web_2.csv":     0,
    "data/capture_streaming_2.csv":      0,
    "data/capture_ai_claude.csv":        1,
    "data/capture_ai_gemini_chatty.csv": 1,
    "data/capture_metro.csv":            0,
    "data/capture_youtube.csv":          0,
    "data/capture_ai_image.csv":         1,
    "data/capture_ai.csv":               1,
    "data/capture_classic_web.csv":      0,
    "data/capture_streaming_3.csv":      0,
}

SERVICE_MAP = {
    "https": 443, "http": 80, "domain": 53, "dns": 53,
    "ntp": 123, "ssh": 22, "ftp": 21, "smtp": 25,
    "imap": 143, "pop3": 110, "quic": 443, "mdns": 5353,
    "ldap": 389, "snmp": 161,
}

dfs = []
for fpath, label in RAW_FILES.items():
    fp = os.path.join(PROJECT_ROOT, fpath)
    if os.path.exists(fp):
        tmp = pd.read_csv(fp)
        tmp["LabelAI"] = label
        dfs.append(tmp)
    else:
        print(f"  [WARN] Fichier manquant : {fpath}")

df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()

# 2. feature engineering - all columns

# --- base numeric columns ---
df["Dur"]    = pd.to_numeric(df["Dur"], errors="coerce")
df["TotPkts"] = pd.to_numeric(df["TotPkts"], errors="coerce")
df["TotBytes"] = pd.to_numeric(df["TotBytes"], errors="coerce")
df = df[df["Dur"] > 0].dropna(subset=["Dur", "TotPkts", "TotBytes"])

df["FlowDuration"]      = df["Dur"]
df["TotalPackets"]      = df["TotPkts"]
df["TotalBytes"]        = df["TotBytes"]
df["FlowBytesPerSec"]   = df["TotBytes"] / df["Dur"]
df["FlowPacketsPerSec"] = df["TotPkts"]  / df["Dur"]
df["AveragePacketSize"] = df["TotBytes"]  / df["TotPkts"]

# --- destination port ---
df["DestinationPort"] = pd.to_numeric(
    df["Dport"].replace(SERVICE_MAP), errors="coerce"
).fillna(0).astype(int)

# --- source port ---
df["SourcePort"] = pd.to_numeric(
    df["Sport"].replace(SERVICE_MAP), errors="coerce"
).fillna(0).astype(int)

# --- protocol ---
# tcp=0, udp=1, arp=2, icmp=3, ipv6-icmp=4, igmp=5, other=6
PROTO_MAP = {"tcp": 0, "udp": 1, "arp": 2, "icmp": 3, "ipv6-icmp": 4, "igmp": 5}
df["ProtocolEncoded"] = df["Proto"].str.lower().map(PROTO_MAP).fillna(6).astype(int)

# --- direction ---
# <-> = established bidirectional = 0
# ->  = outbound unidirectional = 1
# <-  = inbound unidirectional = 2
# <?>  = partial bidirectional = 3
# ?>  = partial outbound = 4
# <?  = partial inbound = 5
# who = broadcast/ARP = 6
DIR_MAP = {"<->": 0, " ->": 1, "<-": 2, "<?>": 3, " ?>": 4, "<?": 5, "who": 6}
df["DirectionEncoded"] = df["Dir"].map(DIR_MAP).fillna(3).astype(int)

# --- connection state ---
# CON = established and active = 0
# FIN = cleanly closed = 1
# RST = reset (force-closed) = 2
# INT = internal/incomplete = 3
# REQ = request only = 4
# RSP = response only = 5
# ACC = accepted = 6
# other = 7
STATE_MAP = {
    "CON": 0, "FIN": 1, "RST": 2, "INT": 3,
    "REQ": 4, "RSP": 5, "ACC": 6, "MHR": 7,
    "EXM": 7, "TIM": 7, "URH": 7,
}
df["StateEncoded"] = df["State"].map(STATE_MAP).fillna(7).astype(int)

# --- cleanup ---
df.replace([np.inf, -np.inf], np.nan, inplace=True)

FEATURES_V7 = [
    # v6 features (numeric)
    "FlowDuration",
    "DestinationPort",
    "TotalPackets",
    "TotalBytes",
    "FlowBytesPerSec",
    "FlowPacketsPerSec",
    "AveragePacketSize",
    # new v7 features (encoded)
    "SourcePort",
    "ProtocolEncoded",
    "DirectionEncoded",
    "StateEncoded",
]
TARGET = "LabelAI"

df.dropna(subset=FEATURES_V7, inplace=True)

# 3. train / val / test split (60/20/20 - same ratio as v6)

train_df, temp_df = train_test_split(df, test_size=0.4, random_state=42, stratify=df[TARGET])
val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df[TARGET])

X_train, y_train = train_df[FEATURES_V7], train_df[TARGET]
X_val,   y_val   = val_df[FEATURES_V7],   val_df[TARGET]
X_test,  y_test  = test_df[FEATURES_V7],  test_df[TARGET]

n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

print("=" * 58)
print("DATASET v7 - TOUTES COLONNES")
print("=" * 58)
print(f"  Total flows  : {len(df)}")
print(f"  AI  (1)      : {(df[TARGET]==1).sum()} ({(df[TARGET]==1).mean()*100:.1f}%)")
print(f"  Non-AI (0)   : {(df[TARGET]==0).sum()} ({(df[TARGET]==0).mean()*100:.1f}%)")
print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")
print(f"  Features     : {len(FEATURES_V7)} (v6 avait 7)")
print(f"\n  Features added: SourcePort, ProtocolEncoded, DirectionEncoded, StateEncoded")
print(f"  Features exclues  : SrcAddr, DstAddr (overfitting), StartTime (pas de pattern temporel)")

# 4. training - LightGBM (best params from v6)

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

y_train_proba = model.predict_proba(X_train)[:, 1]
y_val_proba   = model.predict_proba(X_val)[:, 1]
y_test_proba  = model.predict_proba(X_test)[:, 1]

auc_train = roc_auc_score(y_train, y_train_proba)
auc_val   = roc_auc_score(y_val,   y_val_proba)
auc_test  = roc_auc_score(y_test,  y_test_proba)
gap       = auc_train - auc_val

print(f"\n  AUC train : {auc_train:.4f}")
print(f"  AUC val   : {auc_val:.4f}  (v6 = 0.7447)")
print(f"  AUC test  : {auc_test:.4f}  (v6 = 0.7262)")
print(f"  Gap       : {gap:.4f}   (v6 = 0.1036)")

# best threshold on val
thresholds = np.linspace(0.05, 0.95, 50)
f1s_val    = [f1_score(y_val, (y_val_proba >= t).astype(int), pos_label=1, zero_division=0)
              for t in thresholds]
best_t     = thresholds[int(np.argmax(f1s_val))]

y_test_bt  = (y_test_proba >= best_t).astype(int)
prec_flow  = precision_score(y_test, y_test_bt, pos_label=1, zero_division=0)
rec_flow   = recall_score(y_test,    y_test_bt, pos_label=1, zero_division=0)
f1_flow    = f1_score(y_test,        y_test_bt, pos_label=1, zero_division=0)
cm_flow    = confusion_matrix(y_test, y_test_bt)

print(f"\n  Seuil optimal : {best_t:.2f}")
print(f"  Flow  Prec={prec_flow:.3f} | Rec={rec_flow:.3f} | F1={f1_flow:.3f}  (v6: P=0.397 R=0.684 F1=0.502)")
print(f"  TP={cm_flow[1,1]}  FP={cm_flow[0,1]}  FN={cm_flow[1,0]}  TN={cm_flow[0,0]}")

# 5. cross-validation

print("\nCross-validation 5-fold...")
X_tv = pd.concat([X_train, X_val])
y_tv = pd.concat([y_train, y_val])
cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_tv, y_tv, cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"  CV AUC : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  (v6 = 0.7404 ± 0.0115)")

# feature importance

print("\n" + "=" * 58)
print("FEATURE IMPORTANCE (LightGBM gain)")
print("=" * 58)
importances = model.feature_importances_
feat_imp = sorted(zip(FEATURES_V7, importances), key=lambda x: -x[1])
for feat, imp in feat_imp:
    bar = "#" * int(imp / max(importances) * 30)
    print(f"  {feat:<25} {imp:>6.0f}  {bar}")

# session aggregation

SESSION_THRESHOLD = 0.40
WINDOW_SEC        = 300

raw_all = df.copy()
raw_all["StartTime_parsed"] = pd.to_datetime(
    raw_all["StartTime"], format="%m/%d.%H:%M:%S.%f", errors="coerce"
)
mask = raw_all["StartTime_parsed"].isna()
if mask.any():
    raw_all.loc[mask, "StartTime_parsed"] = pd.to_datetime(
        raw_all.loc[mask, "StartTime"], errors="coerce"
    )
raw_all["ts"] = raw_all["StartTime_parsed"].astype(np.int64) // 10**9

raw_all["proba_ai"] = model.predict_proba(raw_all[FEATURES_V7])[:, 1]
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

print("\n" + "=" * 58)
print(f"SESSION AGGREGATION (window={WINDOW_SEC//60}min, threshold={SESSION_THRESHOLD*100:.0f}%)")
print("=" * 58)
print(f"  Precision : {prec_s:.3f}  (v6 = 0.644)")
print(f"  Recall    : {rec_s:.3f}  (v6 = 0.892)")
print(f"  F1        : {f1_s:.3f}  (v6 = 0.748)")
print(f"  TP={cm_s[1,1]}  FP={cm_s[0,1]}  FN={cm_s[1,0]}  TN={cm_s[0,0]}  (v6: FP=70)")

# plots

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Shadow AI Detection v7 - Toutes colonnes (11 features)", fontsize=14, fontweight="bold")

# 8.1 feature importance
ax = axes[0, 0]
feat_names = [f for f, _ in feat_imp]
feat_vals  = [v for _, v in feat_imp]
colors_imp = ["darkorange" if f in ["SourcePort","ProtocolEncoded","DirectionEncoded","StateEncoded"]
              else "steelblue" for f in feat_names]
ax.barh(feat_names[::-1], feat_vals[::-1], color=colors_imp[::-1], alpha=0.85)
ax.set_title("Feature Importance (gain)")
ax.set_xlabel("Importance")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="darkorange", label="Nouvelles (v7)"),
    Patch(color="steelblue",  label="Existantes (v6)"),
], fontsize=8)

# 8.2 v6 vs v7 comparison (flow-level)
ax = axes[0, 1]
categories = ["AUC test", "F1 flow", "Precision", "Recall"]
v6_vals = [0.726, 0.502, 0.397, 0.684]
v7_vals = [auc_test, f1_flow, prec_flow, rec_flow]
x = np.arange(len(categories))
w = 0.35
b1 = ax.bar(x - w/2, v6_vals, w, label="v6 (7 features)",  color="steelblue",   alpha=0.8)
b2 = ax.bar(x + w/2, v7_vals, w, label="v7 (11 features)", color="darkorange", alpha=0.8)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_title("Flow-level : v6 vs v7")
ax.legend(fontsize=8)

# 8.3 session-level comparison
ax = axes[0, 2]
categories_s = ["Precision", "Recall", "F1", "FP (normalized)"]
v6_s = [0.644, 0.892, 0.748, 70/150]
v7_s = [prec_s, rec_s, f1_s, cm_s[0,1]/150]
x = np.arange(len(categories_s))
b1 = ax.bar(x - w/2, v6_s, w, label="v6", color="steelblue",   alpha=0.8)
b2 = ax.bar(x + w/2, v7_s, w, label="v7", color="darkorange", alpha=0.8)
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(categories_s, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title("Session-level : v6 vs v7")
ax.legend(fontsize=8)

# 8.4 flow confusion matrix
ax = axes[1, 0]
sns.heatmap(cm_flow, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-AI","AI"], yticklabels=["Non-AI","AI"],
            ax=ax, annot_kws={"size":12})
ax.set_title(f"Confusion Matrix - Flow (seuil={best_t:.2f})\nFP={cm_flow[0,1]}")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

# 8.5 session confusion matrix
ax = axes[1, 1]
sns.heatmap(cm_s, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Non-AI","AI"], yticklabels=["Non-AI","AI"],
            ax=ax, annot_kws={"size":12})
ax.set_title(f"Confusion Matrix - Session (seuil=40%)\nFP={cm_s[0,1]}")
ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")

# 8.6 ROC curve
ax = axes[1, 2]
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"v7 AUC={auc_test:.3f}")
ax.fill_between(fpr, tpr, alpha=0.07, color="darkorange")
ax.axhline(0, color="k"); ax.axvline(0, color="k")
ax.plot([0,1],[0,1],"k--",alpha=0.4)
ax.set_title("Courbe ROC (test)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

plt.tight_layout()
out_png = os.path.join(OUT_DIR, "shadow_ai_v7_results.png")
plt.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_png}")

# final summary

delta_auc = auc_test - 0.726
delta_f1  = f1_flow  - 0.502
delta_fps = 70 - cm_s[0,1]
delta_f1s = f1_s - 0.748

print("\n" + "=" * 58)
print("SUMMARY v7 vs v6")
print("=" * 58)
print(f"  Features     : 7 -> 11 (+SourcePort, Protocol, Direction, State)")
print(f"  CV AUC       : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  AUC test     : {auc_test:.4f}  (v6=0.726)  delta={delta_auc:+.4f}")
print(f"  Gap          : {gap:.4f}   (v6=0.104)  {'OK' if gap < 0.12 else 'ATTENTION overfitting'}")
print(f"  F1 flow      : {f1_flow:.4f}  (v6=0.502)  delta={delta_f1:+.4f}")
print(f"  F1 session   : {f1_s:.4f}  (v6=0.748)  delta={delta_f1s:+.4f}")
print(f"  FP session   : {cm_s[0,1]}      (v6=70)     delta={-delta_fps:+d}")

if auc_test > 0.726 or f1_s > 0.748:
    print(f"\n  v7 improves over v6.")
else:
    print(f"\n  No significant improvement over v6.")

joblib.dump(model, os.path.join(OUT_DIR, "shadow_ai_model_v7.pkl"))
print(f"\nModel saved: Model/Version_7/shadow_ai_model_v7.pkl")