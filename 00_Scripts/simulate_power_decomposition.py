import pandas as pd
import numpy as np
from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import plotly.graph_objects as go

def generate_max_production_estimator(systemMetadata):
    latitude = systemMetadata['metadata']['loc_latitude']
    longitude = systemMetadata['metadata']['loc_longitude']
    altitude = systemMetadata['metadata']['loc_altitude']
    Wp_Tot = systemMetadata['metadata']['pv_kwp'] * 1000
    loss = systemMetadata['metadata']['loss'] * 100

    arrays = []
    for array_num, arrayData in systemMetadata['arrays'].items():
        array = Array(
            mount=FixedMount(surface_tilt=arrayData['pv_tilt'], surface_azimuth=arrayData['pv_azimut'], racking_model='open_rack'),
            module_parameters={'pdc0': arrayData['pv_wp'], 'gamma_pdc': -0.004},
            module_type='glass_polymer',
            modules_per_string=arrayData['pv_number'],
            strings=1,
            temperature_model_parameters=TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer'],
        )
        arrays.append(array)

    location = Location(latitude=latitude, longitude=longitude, altitude=altitude, tz='Europe/Zurich')
    system = PVSystem(arrays=arrays,
                      inverter_parameters={'pdc0': Wp_Tot, 'eta_inv_nom': 0.96},
                      losses_parameters={'nameplate_rating': loss, 'soiling': 0, 'shading': 0, 'snow': 0, 'mismatch': 0, 'wiring': 0, 'connections': 0, 'lid': 0, 'age': 0, 'availability': 0})
    modelChain = ModelChain(system, location, clearsky_model='ineichen', aoi_model='no_loss', spectral_model="no_loss", losses_model='pvwatts')

    return modelChain


systemMetadata = {'metadata': {'pv_kwp': 11.84,
  'loc_latitude': 47.0603,
  'pv_pv_type': 'Angebaut',
  'loc_longitude': 7.61508,
  'loc_altitude': 530.6,
  'loss': 0},
 'arrays': {'1': {'pv_number': 1,
   'pv_azimut': 270,
   'pv_manufacturer': 'Suntech',
   'pv_tilt': 35,
   'pv_type': 'STP 310S-20/Wfhb',
   'pv_wp': 180}}}

estimator = generate_max_production_estimator(systemMetadata)

date = pd.to_datetime('2024-07-21')
times = pd.date_range(start=date, end=date + pd.Timedelta(days=1), freq='10min', tz=estimator.location.tz, inclusive='left')
weatherClearSky = estimator.location.get_clearsky(times)  # In W/m2
weatherClearSky['dni'][84:90] = weatherClearSky['dni'][84] * 0.8
# TODO adjust the clear sky model to take into account the horizon https://pvlib-python.readthedocs.io/en/stable/gallery/shading/plot_simple_irradiance_adjustment_for_horizon_shading.html
estimator.run_model(weatherClearSky)
cc_production = estimator.results.ac  # Convert W to kW

# create a series in the lensh and with the same index as cc_production, with random value between 0 and 1
r_factors = pd.Series(np.random.rand(len(cc_production)), index=cc_production.index)
# smooth the random factors with a rolling mean
r_factors = r_factors.rolling(window=3, center=True).mean()
r_production = cc_production * r_factors

# create a series in the lensh and with the same index as cc_production, with 1 every where
a_factors = pd.Series(0.9 * np.ones(len(cc_production)), index=cc_production.index)
# set the 20 last value to 0.5
a_factors.iloc[-50:] = 0.5
a_production = r_production * a_factors

anomalies = (r_production - a_production) / r_production * 100
# plot cc_production, r_production and a_production, filling the area under the value with a solid color

fig = go.Figure()
fig.add_trace(go.Scatter(x=cc_production.index, y=weatherClearSky['dni'], mode='lines', name='Solar Direct Normal Power', fill='tozeroy', fillcolor='rgba(96, 235, 96, 1)', line=dict(color='rgba(96, 235, 96, 1)')))
fig.add_trace(go.Scatter(x=cc_production.index, y=cc_production, mode='lines', name='Optimal System-Specific Power', fill='tozeroy', fillcolor='rgba(66, 216, 227, 1)', line=dict(color='rgba(66, 216, 227, 1)')))
fig.add_trace(go.Scatter(x=r_production.index, y=r_production, mode='lines', name='Power with Regional Factors (clouds, temperature, etc.)', fill='tozeroy', fillcolor='rgba(255, 211, 88, 1)', line=dict(color='rgba(255, 211, 88, 1)')))
fig.add_trace(go.Scatter(x=a_production.index, y=a_production, mode='lines', name='Measured Power (with anomlies)', fill='tozeroy', fillcolor='rgba(227, 66, 89, 1)', line=dict(color='rgba(227, 66, 89, 0)')))
fig.add_trace(go.Scatter(x=a_production.index, y=anomalies, mode='lines', name='Impact of anomalies (in percent)', line=dict(color='red')))

# set the size to 666x1000
fig.update_layout(width=1000, height=666)
# set the y axis to Power [W]
fig.update_yaxes(title_text="Power [W/m2] and Percent [%]")

# put the legend inside the graph
fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

# set max y value to 100
fig.update_yaxes(range=[0, 80])
fig.show()
