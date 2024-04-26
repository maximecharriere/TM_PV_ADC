# Follow https://pvlib-python.readthedocs.io/en/stable/user_guide/introtutorial.html
# # %% Setup
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

# %% Plot the data
# Plot the data in different trace on the same figure
fig = go.Figure()
for device, df in dfs.items():
    fig.add_trace(go.Scatter(x=df['Timestamp'], y=df['energy_daily_norm'], name=device, mode='markers', visible='legendonly'))
    # fig.add_trace(go.Bar(x=df['Timestamp'], y=df['energy_daily_norm'], name=device, width=24 * 60 * 60 * 1000 * 0.8, visible='legendonly'))


fig.update_layout(title='Superposition of Daily Data', xaxis_title='Timestamp', yaxis_title='Energy')
fig.show()
# %%
