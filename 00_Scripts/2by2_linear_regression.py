# %%
# import
## ------------- ##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import plotly.io as pio
from tqdm import tqdm
import requests
import seaborn as sns
from sklearn.linear_model import LinearRegression

# %%
# parameters
## ------------- ##

pio.renderers.default = "browser"  # render plotly figures in browser

PARENT_DATA_DIR = os.getenv('PARENT_DATA_DIR')
if PARENT_DATA_DIR is None:
    raise ValueError("PARENT_DATA_DIR environment variable is not set")


dataDirpath = PARENT_DATA_DIR + r"\PRiOT\dataExport_1"  # "/Applications/Documents/TM Maxime/dataExport_3400_daily"#
dataCacheDirpath = os.path.join(dataDirpath, "cache")
logsDirpath = "../logs"
useCached = False
forceTrain = True
tuneMaxProductionEstimators = True
random_state = 42

minTrainingDays = 30
testingDays = 100

if not os.path.exists(logsDirpath):
    os.makedirs(logsDirpath)

if not os.path.exists(dataCacheDirpath):
    os.makedirs(dataCacheDirpath)

# %%
# functions
## ------------- ##

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

    # Get altitude from LV95 coordinates
    altitude_url = "https://api3.geo.admin.ch/rest/services/height"
    params_altitude = {
        "easting": lv95_easting,
        "northing": lv95_northing
    }

    response_altitude = requests.get(altitude_url, params=params_altitude)
    if response_altitude.status_code != 200:
        raise Exception("Error retrieving altitude: " + response_altitude.text)

    altitude_data = response_altitude.json()
    altitude = altitude_data["height"]

    return float(altitude)


def remove_system(systemName, message):
    if 'systemsName_Valid' in globals() and systemName in systemsName_Valid:
        systemsName_Valid.remove(systemName)
    if 'systemsName_Target' in globals() and systemName in systemsName_Target:
        systemsName_Target.remove(systemName)
    if 'systemsData_EstimatedMaxDailyEnergy' in globals() and systemName in systemsData_EstimatedMaxDailyEnergy.columns:
        systemsData_EstimatedMaxDailyEnergy.drop(columns=systemName, inplace=True)
    if 'systemsData_MeasuredDailyEnergy_train' in globals() and systemName in systemsData_MeasuredDailyEnergy_train.columns:
        systemsData_MeasuredDailyEnergy_train.drop(columns=systemName, inplace=True)
    print(message)

# %%
# import metadata
## ------------- ##
metadataFilepath = os.path.join(dataDirpath, "metadata.json")

with open(metadataFilepath, 'r') as f:
    systemsMetadata = json.load(f)

# Add altitude to metadata, if not already present (TODO : imporove with multi threading)

for systemId, systemMetadata in tqdm(systemsMetadata.items()):
    if "loc_altitude" not in systemMetadata['metadata']:
        if "loc_longitude" in systemMetadata['metadata'] and "loc_latitude" in systemMetadata['metadata']:
            systemMetadata['metadata']["loc_altitude"] = get_altitude_from_wgs84(systemMetadata['metadata']["loc_longitude"], systemMetadata['metadata']["loc_latitude"])

# Split arrays in dictionaries by module number
for systemId, systemMetadata in systemsMetadata.items():
    arrays = {}
    keys_to_delete = []
    for key, value in systemMetadata['metadata'].items():
        if 'mod' in key:
            # Extract the module number
            array_num = key.split('_')[1][-1]
            # Remove the module number from the key
            new_key = '_'.join(key.split('_')[:1] + key.split('_')[2:])
            # Add the key-value pair to the appropriate module dictionary
            if 'arrays' not in systemMetadata:
                systemMetadata['arrays'] = {}
            if array_num not in systemMetadata['arrays']:
                systemMetadata['arrays'][array_num] = {}
            systemMetadata['arrays'][array_num][new_key] = value
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del systemMetadata['metadata'][key]

# Save metadata with altitude
with open(metadataFilepath, 'w') as f:
    json.dump(systemsMetadata, f, indent=4)


# %%
# import data
## ------------- ##   
cacheFilename_systemsData_MeasuredDailyEnergy = os.path.join(dataCacheDirpath, 'systemsData_MeasuredDailyEnergy.pkl')
if useCached and os.path.exists(cacheFilename_systemsData_MeasuredDailyEnergy):
    print(f"Loading cached data in {cacheFilename_systemsData_MeasuredDailyEnergy}")
    systemsData_MeasuredDailyEnergy = pd.read_pickle(cacheFilename_systemsData_MeasuredDailyEnergy)
    systemsName_Valid = systemsData_MeasuredDailyEnergy.columns
else:
    # Load all csv files from the data directory
    systemsData = {}
    for file in os.listdir(dataDirpath):
        if file.endswith(".csv"):
            systemName = file.split("_")[0]
            systemsData[systemName] = pd.read_csv(os.path.join(dataDirpath, file))
            systemsData[systemName]['Datetime'] = pd.to_datetime(systemsData[systemName]['Timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Zurich')
            systemsData[systemName]['Date'] = (systemsData[systemName]['Datetime'] + pd.Timedelta(hours=1)).dt.date  # Convert the datetime to only the date, as the production is the daily production. The +1h is to manage the saving time. Normally PRiOT exports the data at midnight (local time) for the day after (e.g. the energy for the July 1st is saved at July 1st 00:00 Europe/Zurich). However it seams that the saving time is not always correctly handled, and sometime the export is done at 23:00 the day before (e.g. the energy for the July 1st is saved at June 30th 23:00 Europe/Zurich). This is why we add 1h to the datetime to be sure to have the correct date.

    systemsName = list(systemsData.keys())

    df_duplicate_list = list()
    for systemName, systemData in systemsData.items():
        # Save duplicate dates to log list, and the in a log file
        duplicates = systemData[systemData['Date'].duplicated(keep=False)]
        if len(duplicates) > 0:
            df_duplicate_list.append(duplicates)

            # Remove duplicate date where tt_forward_active_energy_total_toDay is the smallest
            # TODO maybe we should sum the energy of the duplicates instead of removing the smallest one. However, when looking in PRiOT Portal, it seams that in the daily energy, only the biggest value is represented. We do the same here.
            systemData.sort_values('tt_forward_active_energy_total_toDay', ascending=True, inplace=True)
            systemsData[systemName].drop_duplicates(subset='Date', keep='last', inplace=True)

        # Set date as the index and sort the data by date
        systemsData[systemName].set_index('Date', inplace=True)
        systemData.sort_index(ascending=True, inplace=True)

    # Save duplicate dates to log file
    df_duplicate = pd.concat(df_duplicate_list)
    print(f"Number of duplicate dates found: {len(df_duplicate)} (see log file for more details)")
    df_duplicate.to_csv(os.path.join(logsDirpath, 'duplicateDates.csv'), index=True)

    ## ----------------------------------------------- ##
    ## Convert data & Filter out invalid PRiOT systems ##
    ## ----------------------------------------------- ##

    systemsName_Valid = systemsName.copy()
    for systemName in systemsName:
        missingData = False
        # Check if the system has measures
        if len(systemsData[systemName]) == 0:
            missingData = True
            print(f"System {systemName} : No measures found")
        # Check if the system has metadata
        if systemName not in systemsMetadata:
            missingData = True
            print(f"System {systemName} : No metadata found")

        else:
            # Check metadata for the system
            for key in ['loc_latitude', 'loc_longitude', 'loc_altitude', 'pv_kwp']:
                # test that the key is present
                if key not in systemsMetadata[systemName]['metadata']:
                    missingData = True
                    print(f"System {systemName} : No '{key}' found")
                # if present, convert the value to a number, if possible
                elif not isinstance(systemsMetadata[systemName]['metadata'][key], (int, float)):
                    try:
                        systemsMetadata[systemName]['metadata'][key] = int(systemsMetadata[systemName]['metadata'][key])
                    except ValueError:
                        try:
                            systemsMetadata[systemName]['metadata'][key] = float(systemsMetadata[systemName]['metadata'][key])
                        except ValueError:
                            missingData = True
                            print(f"System {systemName} : The key-value '{key}:{systemsMetadata[systemName]['metadata'][key]}' is not a number")

            # Check metadata for the arrays
            if 'arrays' not in systemsMetadata[systemName] or len(systemsMetadata[systemName]['arrays']) == 0:
                print(f"System {systemName} : No PV arrays found")
                missingData = True
            else:
                for array_num, arrayData in systemsMetadata[systemName]['arrays'].items():
                    for key in ['pv_tilt', 'pv_azimut', 'pv_wp', 'pv_number']:
                        if key not in arrayData:
                            missingData = True
                            print(f"System {systemName} : No '{key}' found for array {array_num}")
                        # test that the value is a number
                        elif not isinstance(arrayData[key], (int, float)):
                            try:
                                arrayData[key] = int(arrayData[key])
                            except ValueError:
                                try:
                                    arrayData[key] = float(arrayData[key])
                                except ValueError:
                                    missingData = True
                                    print(f"System {systemName} : The key-value '{key}:{arrayData[key]}' is not a number for array {array_num}")

            # add the loss metadata if not present
            if 'loss' not in systemsMetadata[systemName]['metadata']:
                systemsMetadata[systemName]['metadata']['loss'] = 0

        if missingData:
            systemsName_Valid.remove(systemName)
            print(f"-> Removing system {systemName} from the list of systems")

    print(f"Number of systems with all the necessary data: {len(systemsName_Valid)}/{len(systemsName)}")

    # # Filter out systems with less than X days of data
    # for systemName in systemsName_Valid[:]:  # Create a copy of the list using slicing [:] to avoid removing elements while iterating over the list itself
    #     if len(systemsData[systemName]) < minMeasurements:
    #         systemsName_Valid.remove(systemName)
    #         print(f"-> Removing system {systemName} from the list of systems because it has less than {minMeasurements} days of data")

    # print(f"Number of systems with at least {minMeasurements} days of data: {len(systemsName_Valid)}/{len(systemsName)}")

    ## ---------------------------------------------------------------------------- ##
    ## Create one 2D DataFrame with the daily production of every remaining systems ##
    ## ---------------------------------------------------------------------------- ##

    # Create an empty list to store all measured data for each systems
    systemsData_MeasuredDailyEnergy_List = []

    # Iterate over each key-value pair in the systemsData dictionary
    for systemName in systemsName_Valid:
        # Extract the 'tt_forward_active_energy_total_toDay' column from the current dataframe
        measuredDailyEnergy = systemsData[systemName]['tt_forward_active_energy_total_toDay']

        # Rename the column with the system name
        measuredDailyEnergy.rename(systemName, inplace=True)

        systemsData_MeasuredDailyEnergy_List.append(measuredDailyEnergy)
        # Concatenate the column to the new_dataframe

    # Concatenate all the columns in the list to create one dataframe
    systemsData_MeasuredDailyEnergy = pd.concat(systemsData_MeasuredDailyEnergy_List, axis=1)
    systemsData_MeasuredDailyEnergy.index = pd.to_datetime(systemsData_MeasuredDailyEnergy.index)
    systemsData_MeasuredDailyEnergy.sort_index(inplace=True)

    ## ------------------ ##
    ## Save the dataframe ##
    ## ------------------ ##
    # Save the dataframe for later use
    # create cache directory if it does not exist

    systemsData_MeasuredDailyEnergy.to_pickle(cacheFilename_systemsData_MeasuredDailyEnergy)

# Print the dataframe
systemsData_MeasuredDailyEnergy


# %%
# filter out outliers
## ------------- ## 

# create a mask to remove outliers of the shape of systemsData_MeasuredDailyEnergy
inlier_masks = pd.DataFrame(True, index=systemsData_MeasuredDailyEnergy.index, columns=systemsData_MeasuredDailyEnergy.columns)
# set the mask to false for value in systemsData_MeasuredDailyEnergy under 1kWh
inlier_masks[systemsData_MeasuredDailyEnergy < 1] = False

# set the mask to false, for each value in the upper quantile (Q3 + 1.5 * IQR) of each column
for systemName in systemsData_MeasuredDailyEnergy.columns:
    Q1 = systemsData_MeasuredDailyEnergy[systemName].quantile(0.25)
    Q3 = systemsData_MeasuredDailyEnergy[systemName].quantile(0.75)
    IQR = Q3 - Q1
    inlier_masks[systemsData_MeasuredDailyEnergy[systemName] > Q3 + 1.5 * IQR] = False

filtered_data = systemsData_MeasuredDailyEnergy.copy()
filtered_data[~inlier_masks] = np.nan

# %%
# normalize data
## ------------- ##

filtered_data /= filtered_data.max()


# %%
# Compute lin regression and r2
## ------------- ## 

# To store the regressors and their evaluation metrics
lin_regressors = {}
r2_scores = pd.Series(index=systemsData_MeasuredDailyEnergy.columns)

for i, systemName in enumerate([name for name in systemsData_MeasuredDailyEnergy.columns if name != 'a001035']):
    val = filtered_data[['a001035', systemName]].dropna()
    if not len(val):
        continue
    X = val[[systemName]]
    y = val['a001035']
    
    # Fit the linear regression model
    linreg = LinearRegression().fit(X, y)
    lin_regressors[systemName] = linreg
    
    # Make predictions   
    r2_scores[systemName] = linreg.score(X, y)


# %%
# Predict using the best lin reg
## ------------- ## 

# Create a list of index in r2_scores where the value is more than 0.9
high_r2_scores = r2_scores[r2_scores > 0.9].index
print("number of systems with R^2 > 0.95:", len(high_r2_scores))


X = filtered_data.drop(columns='a001035')
y = filtered_data['a001035']

# Remove observations where y is NA
X = X[~y.isna()]
y = y[~y.isna()]

# Create an empty DataFrame to store the predictions
inter_pred = pd.DataFrame(index=y.index, columns=high_r2_scores)

for systemName in high_r2_scores:
    if systemName not in lin_regressors:
        continue
    
    # Drop observations with NA values in the current feature
    valid_X = X[[systemName]].dropna()
    
    # Get the indices of valid observations
    valid_indices = valid_X.index
    
    # Make predictions only for valid observations
    inter_pred.loc[valid_indices, systemName] = lin_regressors[systemName].predict(valid_X)

# Display the inter_pred DataFrame
inter_pred

# For each day (index) in the inter_pred DataFrame, calculate the mean of the predictions
pred = inter_pred.mean(axis=1)



# %%
# Plot
## ------------- ## 

# Create a figure with 52 subplots
fig, axs = plt.subplots(10, 5, figsize=(20,30))

# Flatten the axs array to iterate over it
axs = axs.flatten()

# Iterate over each system
for i, systemName in enumerate([name for name in filtered_data.columns  if name != 'a001035']):
    if systemName not in lin_regressors:
        continue
    # Plot the daily production of the current system against the daily production of all other systems
    val = filtered_data[['a001035', systemName]].dropna()
    X = val[[systemName]]
    y = val['a001035']


    # Plot all
    sns.scatterplot(x=X.values.flatten(), y=y, ax=axs[i])
    # Plot inliers
    # sns.scatterplot(x=X[inlier_mask].values.flatten(), y=y[inlier_mask], ax=axs[i], color='blue', label='Inliers')
    # # Plot outliers
    # sns.scatterplot(x=X[outlier_mask].values.flatten(), y=y[outlier_mask], ax=axs[i], color='red', label='Outliers')
    
    # # Plot the regression line
    line_x = pd.DataFrame(np.linspace(X.min().iloc[0], X.max().iloc[0], 10), columns=[systemName])
    line_y = lin_regressors[systemName].predict(line_x)
    axs[i].plot(line_x, line_y, color='green', linewidth=2)
    
    axs[i].set_xlabel(systemName)
    axs[i].set_ylabel('a001035')
    # set title color to red if in the high_r2_scores list
    axs[i].set_title(f'R2: {r2_scores[systemName]:.2f}', color='green' if systemName in high_r2_scores else 'red')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig("2by2_linear_regression.png")
# %%
# Do the same with a SplittedLinearRegression class. Not working yet

# from sklearn.linear_model import LinearRegression, RANSACRegressor

# class SplittedLinearRegression:
#     def __init__(self, min_score=0.9):
#         self.regressors = None
#         self.scores = None
#         self.min_score = min_score
#     def fit(self, X, y):
#         self.regressors = {}
#         self.scores = pd.Series(index=X.columns)
#         # fit a linear regression for each feature in X
#         for featureName in X.columns:
#             # Manage NA value: Keep only observations where both systems (X and y) have values
#             val = pd.concat([X[featureName], y], axis=1).dropna()
#             if not len(val):
#                 continue
#             X_val = val[[featureName]]
#             y_val = val[y]

#             # Fit the linear regression model
#             linreg = LinearRegression().fit(X_val, y_val)
#             self.regressors[featureName] = linreg

#             # Make predictions   
#             self.scores[featureName] = linreg.score(X_val, y_val)

#     def predict(self, X):
#         # Make predictions for each features in X where the score is above the threshold min_score
#         high_scores_features = self.scores[self.scores > self.min_score].index

#         # Create an empty DataFrame to store the intermediate predictions
#         inter_y_pred = pd.DataFrame(index=X.index, columns=high_scores_features)
#         for featureName in high_scores_features:
#             if featureName not in self.regressors:
#                 continue
#             # Drop observations with NA values in the current feature
#             valid_X = X[[featureName]].dropna()
#             # Get the indices of valid observations
#             valid_indices = valid_X.index
#             # Make predictions only for valid observations
#             inter_y_pred.loc[valid_indices, featureName] = self.regressors[featureName].predict(valid_X)
#         # Return the mean of the predictions
#         return inter_y_pred.mean(axis=1)
    

# # Create a list of index in r2_scores where the value is more than 0.9
# high_r2_scores = r2_scores[r2_scores > 0.9].index
# print("number of systems with R^2 > 0.95:", len(high_r2_scores))
# # Prepare the feature matrix X and the target vector y
# X = filtered_data.drop(columns='a001035')
# y = filtered_data['a001035']

# # Remove observations where y is NA
# X = X[~y.isna()]
# y = y[~y.isna()]

# # Create an empty DataFrame to store the predictions
# inter_pred = pd.DataFrame(index=y.index, columns=high_r2_scores)

# for systemName in high_r2_scores:
#     if systemName not in lin_regressors:
#         continue
    
#     # Drop observations with NA values in the current feature
#     valid_X = X[[systemName]].dropna()
    
#     # Get the indices of valid observations
#     valid_indices = valid_X.index
    
#     # Make predictions only for valid observations
#     inter_pred.loc[valid_indices, systemName] = lin_regressors[systemName].predict(valid_X)

# # Display the inter_pred DataFrame
# inter_pred

# # For each day (index) in the inter_pred DataFrame, calculate the mean of the predictions
# pred = inter_pred.mean(axis=1)

# # plot the predictions against the target, with markers, using plotly
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=y.index, y=y, mode='markers', name='True'))
# fig.add_trace(go.Scatter(x=pred.index, y=pred, mode='markers', name='Predicted'))
# fig.show()
