import pandas as pd
import requests


def get_altitude_from_lv95(easting, northing):
    # Get altitude from LV95 coordinates
    altitude_url = "https://api3.geo.admin.ch/rest/services/height"
    params_altitude = {
        "easting": easting,
        "northing": northing
    }

    response_altitude = requests.get(altitude_url, params=params_altitude)
    if response_altitude.status_code != 200:
        raise Exception("Error retrieving altitude: " + response_altitude.text)

    altitude_data = response_altitude.json()
    altitude = altitude_data["height"]

    return float(altitude)

def get_altitude_from_wgs84(longitude, latitude):
    # Convert WGS84 to LV95
    lv95_url = "https://geodesy.geo.admin.ch/reframe/wgs84tolv95"
    params_lv95 = {
        "easting": longitude,
        "northing": latitude,
        "format": "json"
    }

    response_lv95 = requests.get(lv95_url, params=params_lv95)
    if response_lv95.status_code != 200:
        raise Exception("Error converting WGS84 to LV95: " + response_lv95.text)

    lv95_data = response_lv95.json()
    lv95_easting = lv95_data["easting"]
    lv95_northing = lv95_data["northing"]

    return get_altitude_from_lv95(lv95_easting, lv95_northing)

# Convert the power production with a given frequency to the total daily energy
def power_to_daily_energy(df_power):
    # check that df_power has a datetime index with a frequency attribute
    if not isinstance(df_power.index, pd.DatetimeIndex):
        raise ValueError("df_power must have a datetime index to compute the daily energy")
    if df_power.index.freq is None:
        raise ValueError("df_power must have a frequency to compute the daily energy")
    
    # Get the frequency in minutes
    freq_in_minutes = pd.Timedelta(df_power.index.freq).seconds / 60
    # Convert power from kW to kWh
    df_energy = df_power * (freq_in_minutes / 60)
    # Resample to daily frequency and sum the values
    daily_energy = df_energy.resample('D').sum()

    return daily_energy