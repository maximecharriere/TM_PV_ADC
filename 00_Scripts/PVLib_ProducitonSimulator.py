# %% Setup
import pandas as pd
import plotly.graph_objects as go
import os
import json

PARENT_DATA_DIR = os.getenv('PARENT_DATA_DIR')
if PARENT_DATA_DIR is None:
    raise ValueError("PARENT_DATA_DIR environment variable is not set")

dataDirpath = PARENT_DATA_DIR + r"\PRiOT\dataExport_3400_daily"
# %% Load metadata
# Load the metadata JSON file
metadataFilepath = os.path.join(dataDirpath, "metadata.json")

with open(metadataFilepath, 'r') as f:
    metadata = json.load(f)

# %% Load data
# Load all csv files from the data directory
dfs = {}
for file in os.listdir(dataDirpath):
    if file.endswith(".csv"):
        deviceName = file.split("_")[0]
        dfs[deviceName] = pd.read_csv(os.path.join(dataDirpath, file))
        dfs[deviceName]['Timestamp'] = pd.to_datetime(dfs[deviceName]['Timestamp'], unit='ms')
        dfs[deviceName]['energy_daily_norm'] = dfs[deviceName]['tt_forward_active_energy_total_toDay'] / metadata[deviceName]['metadata']['pv_kwp']
