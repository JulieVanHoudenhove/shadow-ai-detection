import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Load the data
capture_ai_heavy = pd.read_csv("data/capture_ai_heavy_2.csv")
capture_ai_leger = pd.read_csv("data/capture_normal_ai_2.csv")
capture_cloud_work = pd.read_csv("data/capture_cloud_work_2.csv")
capture_normal_web= pd.read_csv("data/capture_normal_web_2.csv")
capture_streaming= pd.read_csv("data/capture_streaming_2.csv")

capture_ai_claude = pd.read_csv("data/capture_ai_claude.csv")
capture_ai_gemini_chatty = pd.read_csv("data/capture_ai_gemini_chatty.csv")
capture_metro = pd.read_csv("data/capture_metro.csv")
capture_youtube = pd.read_csv("data/capture_youtube.csv")

capture_ai_image = pd.read_csv("data/capture_ai_image.csv")
capture_ai = pd.read_csv("data/capture_ai.csv")
capture_classic_web = pd.read_csv("data/capture_classic_web.csv")
capture_streaming_3 = pd.read_csv("data/capture_streaming_3.csv")

# Labeling
normal = pd.concat([capture_cloud_work, capture_normal_web, capture_streaming, capture_metro, capture_youtube, capture_classic_web, capture_streaming_3])
ai = pd.concat([capture_ai_heavy, capture_ai_leger, capture_ai_claude, capture_ai_gemini_chatty, capture_ai_image, capture_ai])

normal["label_ia"] = 0
ai["label_ia"] = 1

supervised_dataset = pd.concat([normal, ai], ignore_index=True)
supervised_dataset.shape

# Strip leading/trailing whitespace from column names
supervised_dataset.columns = supervised_dataset.columns.str.strip()
# Compute derived columns for the dataset
supervised_dataset["FlowBytesPerSec"] = supervised_dataset["TotBytes"] / supervised_dataset["Dur"]
supervised_dataset["FlowPacketsPerSec"] = supervised_dataset["TotPkts"] / supervised_dataset["Dur"]
supervised_dataset["AveragePacketSize"] = supervised_dataset["TotBytes"] / supervised_dataset["TotPkts"]

supervised_dataset = supervised_dataset.rename(
    columns={
        "Dur":"FlowDuration",
        "Proto":"Protocol",
        "Dport":"DestinationPort",
        "TotPkts":"TotalPackets",
        "TotBytes":"TotalBytes",
        "State":"ConnectionState",
        "Dir":"Direction",
        "label_ia":"LabelAI"
    }
)
# Keep only the relevant columns
features = [
    "FlowDuration",
    "DestinationPort",
    "TotalPackets",
    "TotalBytes",
    "FlowBytesPerSec",
    "FlowPacketsPerSec",
    "AveragePacketSize",
    "LabelAI"
]
supervised_dataset = supervised_dataset[features]

# Exclude zero-duration flows to avoid division by zero
supervised_dataset = supervised_dataset[supervised_dataset["FlowDuration"] >= 0]
supervised_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
supervised_dataset.dropna(inplace=True)
# Normalize port names to numeric values
supervised_dataset["DestinationPort"] = supervised_dataset["DestinationPort"].replace({
    "https":   443,
    "http":    80,
    "domain":  53,
    "dns":     53,
    "ntp":     123,
    "ssh":     22,
    "ftp":     21,
    "smtp":    25,
    "imap":    143,
    "pop3":    110,
    "ldap":    389,
    "snmp":    161,
    "quic":    443,
    "mdns":    5353,
})

# Train / test / validation split
supervised_train, temporary_subset = train_test_split(supervised_dataset, test_size=0.4, random_state=42)
supervised_validation, supervised_test = train_test_split(temporary_subset, test_size=0.5, random_state=42)