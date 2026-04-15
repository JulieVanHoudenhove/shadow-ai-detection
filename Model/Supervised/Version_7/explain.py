# explainability for sessions flagged as AI
# SHAP to understand which features drove the decision
# output: detection_log.csv and .txt
# usage: python3 explain.py  (or pass a csv file to run on new data)

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import shap
from datetime import datetime
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", ".."))
OUT_DIR      = os.path.dirname(__file__)

# 1. load v7 model

MODEL_PATH = os.path.join(OUT_DIR, "shadow_ai_model_v7.pkl")
if not os.path.exists(MODEL_PATH):
    print(f"[ERROR] Model not found: {MODEL_PATH}")
    print("  Run Train_v7.py first to train the model.")
    sys.exit(1)

model = joblib.load(MODEL_PATH)
print(f"Model loaded: shadow_ai_model_v7.pkl")

FEATURES = [
    "FlowDuration", "DestinationPort", "TotalPackets",
    "TotalBytes", "FlowBytesPerSec", "FlowPacketsPerSec", "AveragePacketSize",
    "SourcePort", "ProtocolEncoded", "DirectionEncoded", "StateEncoded",
]

# human-readable feature labels
FEATURE_LABELS = {
    "FlowDuration":      "flow duration",
    "DestinationPort":   "port destination",
    "TotalPackets":      "nb paquets",
    "TotalBytes":        "nb octets",
    "FlowBytesPerSec":   "throughput (bytes/s)",
    "FlowPacketsPerSec": "throughput (packets/s)",
    "AveragePacketSize": "taille moy. paquet",
    "SourcePort":        "port source",
    "ProtocolEncoded":   "protocole",
    "DirectionEncoded":  "direction flux",
    "StateEncoded":      "connection state",
}

PROTO_DECODE = {0:"TCP", 1:"UDP", 2:"ARP", 3:"ICMP", 4:"IPv6-ICMP", 5:"IGMP", 6:"autre"}
DIR_DECODE   = {0:"bidirectionnel", 1:"sortant", 2:"entrant", 3:"partiel", 4:"partiel-out", 5:"partiel-in", 6:"broadcast"}
STATE_DECODE = {0:"CON (established)", 1:"FIN (closed)", 2:"RST (reset)", 3:"INT (internal)",
                4:"REQ (request)", 5:"RSP (response)", 6:"ACC (accepted)", 7:"other"}

# decision threshold (same as Train_v7)
FLOW_THRESHOLD    = 0.55
SESSION_THRESHOLD = 0.40
WINDOW_SEC        = 300

# load data

SERVICE_MAP = {
    "https": 443, "http": 80, "domain": 53, "dns": 53,
    "ntp": 123, "ssh": 22, "ftp": 21, "smtp": 25,
    "imap": 143, "pop3": 110, "quic": 443, "mdns": 5353,
    "ldap": 389, "snmp": 161,
}

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
    "data/capture_ai.csv":              1,
    "data/capture_classic_web.csv":      0,
    "data/capture_streaming_3.csv":      0,
}

if len(sys.argv) > 1:
    # external file passed as argument
    custom_path = sys.argv[1]
    if not os.path.exists(custom_path):
        custom_path = os.path.join(PROJECT_ROOT, sys.argv[1])
    df_raw = pd.read_csv(custom_path)
    df_raw.columns = df_raw.columns.str.strip()
    df_raw["LabelAI"] = -1   # inconnu
    print(f"Fichier externe : {custom_path} ({len(df_raw)} lignes)")
    dfs = [df_raw]
else:
    dfs = []
    for fpath, label in RAW_FILES.items():
        fp = os.path.join(PROJECT_ROOT, fpath)
        if os.path.exists(fp):
            tmp = pd.read_csv(fp)
            tmp["LabelAI"] = label
            dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.strip()

# feature engineering (same as Train_v7)
df["Dur"]     = pd.to_numeric(df["Dur"],     errors="coerce")
df["TotPkts"] = pd.to_numeric(df["TotPkts"], errors="coerce")
df["TotBytes"]= pd.to_numeric(df["TotBytes"],errors="coerce")
df = df[df["Dur"] > 0].dropna(subset=["Dur","TotPkts","TotBytes"])

df["FlowDuration"]      = df["Dur"]
df["TotalPackets"]      = df["TotPkts"]
df["TotalBytes"]        = df["TotBytes"]
df["FlowBytesPerSec"]   = df["TotBytes"] / df["Dur"]
df["FlowPacketsPerSec"] = df["TotPkts"]  / df["Dur"]
df["AveragePacketSize"] = df["TotBytes"]  / df["TotPkts"]
df["DestinationPort"]   = pd.to_numeric(df["Dport"].replace(SERVICE_MAP), errors="coerce").fillna(0).astype(int)
df["SourcePort"]        = pd.to_numeric(df["Sport"].replace(SERVICE_MAP), errors="coerce").fillna(0).astype(int)

PROTO_MAP = {"tcp":0,"udp":1,"arp":2,"icmp":3,"ipv6-icmp":4,"igmp":5}
DIR_MAP   = {"<->":0," ->":1,"<-":2,"<?>":3," ?>":4,"<?":5,"who":6}
STATE_MAP = {"CON":0,"FIN":1,"RST":2,"INT":3,"REQ":4,"RSP":5,"ACC":6,
             "MHR":7,"EXM":7,"TIM":7,"URH":7}

df["ProtocolEncoded"]  = df["Proto"].str.lower().map(PROTO_MAP).fillna(6).astype(int)
df["DirectionEncoded"] = df["Dir"].map(DIR_MAP).fillna(3).astype(int)
df["StateEncoded"]     = df["State"].map(STATE_MAP).fillna(7).astype(int)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=FEATURES, inplace=True)
df = df.reset_index(drop=True)

print(f"Flows loaded: {len(df)}")

# 3. prediction + SHAP

X = df[FEATURES]
probas     = model.predict_proba(X)[:, 1]
df["proba_ai"] = probas
df["pred_ai"]  = (probas >= FLOW_THRESHOLD).astype(int)

print("Calcul SHAP values...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
# binary LightGBM: shap_values can be a list [class0, class1]
if isinstance(shap_values, list):
    shap_ai = shap_values[1]
else:
    shap_ai = shap_values
print("SHAP computed.")

# session aggregation

df["StartTime_parsed"] = pd.to_datetime(df["StartTime"], format="%m/%d.%H:%M:%S.%f", errors="coerce")
mask = df["StartTime_parsed"].isna()
if mask.any():
    df.loc[mask, "StartTime_parsed"] = pd.to_datetime(df.loc[mask, "StartTime"], errors="coerce")
df["ts"] = df["StartTime_parsed"].astype(np.int64) // 10**9
df["session_window"] = df["ts"] // WINDOW_SEC

sessions = df.groupby(["SrcAddr", "session_window"]).agg(
    n_flows       = ("pred_ai", "count"),
    n_ai_flows    = ("pred_ai", "sum"),
    true_label    = ("LabelAI", "max"),
    mean_proba    = ("proba_ai", "mean"),
    total_bytes   = ("TotalBytes", "sum"),
    mean_duration = ("FlowDuration", "mean"),
    indices       = ("pred_ai", lambda x: list(x.index)),
).reset_index()

sessions["pct_ai_flows"]    = sessions["n_ai_flows"] / sessions["n_flows"]
sessions["session_pred_ai"] = (sessions["pct_ai_flows"] >= SESSION_THRESHOLD).astype(int)

# generate explanations

def format_feature_value(feat, val):
    """Retourne une valeur lisible selon la feature."""
    if feat == "ProtocolEncoded":
        return PROTO_DECODE.get(int(val), str(val))
    if feat == "DirectionEncoded":
        return DIR_DECODE.get(int(val), str(val))
    if feat == "StateEncoded":
        return STATE_DECODE.get(int(val), str(val))
    if feat in ("DestinationPort", "SourcePort"):
        return f"port {int(val)}"
    if feat in ("FlowBytesPerSec", "FlowPacketsPerSec"):
        return f"{val:.1f}/s"
    if feat == "AveragePacketSize":
        return f"{val:.0f} octets"
    if feat == "FlowDuration":
        return f"{val:.3f}s"
    if feat in ("TotalPackets", "TotalBytes"):
        return f"{int(val)}"
    return f"{val:.2f}"

def explain_session(session_row, df_flows, shap_matrix):
    """
    Génère une explication textuelle pour une session flaggée AI.
    Retourne un dict avec les infos de log.
    """
    idx_list = session_row["indices"]

    # SHAP averaged over AI flows in the session
    ai_idx   = [i for i in idx_list if df_flows.loc[i, "pred_ai"] == 1]
    if not ai_idx:
        ai_idx = idx_list   # fallback : tous les flows

    shap_mean = np.abs(shap_matrix[ai_idx]).mean(axis=0)
    top_idx   = np.argsort(shap_mean)[::-1][:3]   # top 3 features

    # build reasons
    reasons = []
    for fi in top_idx:
        feat  = FEATURES[fi]
        label = FEATURE_LABELS[feat]
        vals  = [df_flows.loc[i, feat] for i in ai_idx]
        val_m = np.mean(vals)
        val_f = format_feature_value(feat, val_m)
        contrib = shap_mean[fi]
        reasons.append(f"{label}={val_f} (contribution SHAP: {contrib:.3f})")

    # confidence score
    mean_proba = session_row["mean_proba"]
    confidence = "HIGH" if mean_proba > 0.80 else "MEDIUM" if mean_proba > 0.65 else "LOW"

    return {
        "src_ip":       session_row["SrcAddr"],
        "window_start": datetime.fromtimestamp(session_row["session_window"] * WINDOW_SEC).strftime("%Y-%m-%d %H:%M:%S"),
        "n_flows":      int(session_row["n_flows"]),
        "n_ai_flows":   int(session_row["n_ai_flows"]),
        "pct_ai":       round(session_row["pct_ai_flows"] * 100, 1),
        "mean_proba":   round(float(mean_proba), 4),
        "confidence":   confidence,
        "true_label":   int(session_row["true_label"]),
        "reason_1":     reasons[0] if len(reasons) > 0 else "",
        "reason_2":     reasons[1] if len(reasons) > 1 else "",
        "reason_3":     reasons[2] if len(reasons) > 2 else "",
    }

# write log

ai_sessions = sessions[sessions["session_pred_ai"] == 1].reset_index(drop=True)
log_entries = []

print(f"\nAI sessions detected: {len(ai_sessions)}")
print("Generating explanations...")

for _, row in ai_sessions.iterrows():
    entry = explain_session(row, df, shap_ai)
    log_entries.append(entry)

# --- CSV ---
log_csv = os.path.join(OUT_DIR, "detection_log.csv")
df_log  = pd.DataFrame(log_entries)
df_log.to_csv(log_csv, index=False)

# --- human-readable TXT ---
log_txt = os.path.join(OUT_DIR, "detection_log.txt")
run_ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(log_txt, "w") as f:
    f.write("=" * 65 + "\n")
    f.write(f"  SHADOW AI DETECTION LOG - v7\n")
    f.write(f"  Generated: {run_ts}\n")
    f.write(f"  Flows analysed: {len(df)} | AI sessions detected: {len(ai_sessions)}\n")
    f.write("=" * 65 + "\n\n")

    for i, entry in enumerate(log_entries, 1):
        true_str = "VRAI POSITIF" if entry["true_label"] == 1 else \
                   "FAUX POSITIF" if entry["true_label"] == 0 else "INCONNU"

        f.write(f"[ALERTE #{i}] {true_str}\n")
        f.write(f"  IP source    : {entry['src_ip']}\n")
        f.write(f"  Window       : {entry['window_start']} (5 min)\n")
        f.write(f"  Flows        : {entry['n_ai_flows']} AI sur {entry['n_flows']} ({entry['pct_ai']}%)\n")
        f.write(f"  Confidence   : {entry['confidence']} (mean probability = {entry['mean_proba']:.2%})\n")
        f.write(f"\n  Why the model flagged this as AI:\n")
        f.write(f"    1. {entry['reason_1']}\n")
        if entry['reason_2']:
            f.write(f"    2. {entry['reason_2']}\n")
        if entry['reason_3']:
            f.write(f"    3. {entry['reason_3']}\n")
        f.write("\n" + "-" * 65 + "\n\n")

    # global metrics (if labels available)
    known = sessions[sessions["true_label"] != -1]
    if len(known) > 0:
        y_true = known["true_label"]
        y_pred = known["session_pred_ai"]
        f.write("=" * 65 + "\n")
        f.write("  GLOBAL METRICS\n")
        f.write("=" * 65 + "\n")
        f.write(f"  Total sessions analysed    : {len(sessions)}\n")
        f.write(f"  AI sessions flagged        : {len(ai_sessions)}\n")
        f.write(f"  Precision  : {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.3f}\n")
        f.write(f"  Recall     : {recall_score(y_true, y_pred,    pos_label=1, zero_division=0):.3f}\n")
        f.write(f"  F1         : {f1_score(y_true, y_pred,        pos_label=1, zero_division=0):.3f}\n")

# console preview

print(f"\n{'='*65}")
print(f"  PREVIEW OF FIRST 5 ALERTS")
print(f"{'='*65}")

for entry in log_entries[:5]:
    true_str = "VRAI POSITIF" if entry["true_label"] == 1 else \
               "FAUX POSITIF" if entry["true_label"] == 0 else "INCONNU"
    print(f"\n  [{true_str}] IP={entry['src_ip']} | {entry['window_start']}")
    print(f"    Flows AI : {entry['n_ai_flows']}/{entry['n_flows']} ({entry['pct_ai']}%) | Confiance : {entry['confidence']} ({entry['mean_proba']:.2%})")
    print(f"    Raisons :")
    print(f"      1. {entry['reason_1']}")
    if entry['reason_2']: print(f"      2. {entry['reason_2']}")
    if entry['reason_3']: print(f"      3. {entry['reason_3']}")

print(f"\n{'='*65}")
print(f"  Full log saved:")
print(f"  {log_csv}")
print(f"  {log_txt}")
print(f"{'='*65}")