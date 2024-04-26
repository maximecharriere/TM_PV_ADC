import pandas as pd
import plotly.graph_objects as go
import os

PARENT_DATA_DIR = os.getenv('PARENT_DATA_DIR')

# Load the CSV file into a DataFrame
# The type are "string,string,long,dateTime:RFC3339,dateTime:RFC3339,dateTime:RFC3339,double,string,string,string"
# Load only 10k first rows
df_P_ac_L1 = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_P_ac_L1_2004.csv",
                         skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_P_ac_L2 = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_P_ac_L2_2004.csv",
                         skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_P_ac_L3 = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_P_ac_L3_2004.csv",
                         skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_G_0 = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_G_0_2004.csv",
                     skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_G_i = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_G_i_2004.csv",
                     skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_I_dc = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_I_dc_2004.csv",
                      skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_I_ref = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_I_ref_2004.csv",
                       skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_T_a = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_T_a_2004.csv",
                     skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_T_gen = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_T_gen_2004.csv",
                       skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_T_ref = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_T_ref_2004.csv",
                       skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)
df_U_dc = pd.read_csv(PARENT_DATA_DIR+r"\Tiergarten_Ost_1PVS_1min\measurements\01_02_Tiergarten_Ost_U_dc+_2004.csv",
                      skiprows=3, parse_dates=['_start', '_time', '_stop'], nrows=70000)

# df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

# # Define start and end dates for filtering
# start_date = pd.to_datetime("2004-06-01T00:00:00Z")
# end_date = pd.to_datetime("2004-06-08T00:00:00Z")

# # Filter the DataFrame based on '_time'
# df_P_ac_L1 = df_P_ac_L1[(df_P_ac_L1['_time'] >= start_date) & (df_P_ac_L1['_time'] <= end_date)]
# df_P_ac_L2 = df_P_ac_L2[(df_P_ac_L2['_time'] >= start_date) & (df_P_ac_L2['_time'] <= end_date)]
# df_P_ac_L3 = df_P_ac_L3[(df_P_ac_L3['_time'] >= start_date) & (df_P_ac_L3['_time'] <= end_date)]


# # Assuming your DataFrame has columns 'timestamp' and 'value'
fig = go.Figure()

# Add a line plot
fig.add_trace(go.Scattergl(
    x=df_P_ac_L1['_time'], y=df_P_ac_L1['_value'], name='L1', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_P_ac_L2['_time'], y=df_P_ac_L2['_value'], name='L2', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_P_ac_L3['_time'], y=df_P_ac_L3['_value'], name='L3', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_G_0['_time'], y=df_G_0['_value'], name='G_0', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_G_i['_time'], y=df_G_i['_value'], name='G_i', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_I_dc['_time'], y=df_I_dc['_value'], name='I_dc', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_I_ref['_time'], y=df_I_ref['_value'], name='I_ref', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_T_a['_time'], y=df_T_a['_value'], name='T_a', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_T_gen['_time'], y=df_T_gen['_value'], name='T_gen', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_T_ref['_time'], y=df_T_ref['_value'], name='T_ref', mode='markers'))
fig.add_trace(go.Scattergl(
    x=df_U_dc['_time'], y=df_U_dc['_value'], name='U_dc', mode='markers'))
# fig.add_trace(go.Scattergl(x=df['Date & Time'], y=df['gen [kW]'], mode='markers', name='gen [kW]'))
# fig.add_trace(go.Scattergl(x=df['Date & Time'], y=df['Grid [kW]'], mode='markers', name='Grid [kW]'))
# fig.add_trace(go.Scattergl(x=df['Date & Time'], y=df['Solar [kW]'], mode='markers', name='Solar [kW]'))
# fig.add_trace(go.Scattergl(x=df['Date & Time'], y=df['Solar+ [kW]'], mode='markers', name='Solar+ [kW]'))


# Update layout to add zooming capabilities
fig.update_layout(
    title="Time Series Visualization",
    xaxis_title="Time",
    yaxis_title="Value",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1d", step="day", stepmode="backward"),
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

# Show the plot
fig.show()
