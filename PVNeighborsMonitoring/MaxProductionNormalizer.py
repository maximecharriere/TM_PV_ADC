import pandas as pd
import numpy as np

from pvlib.pvsystem import PVSystem, Array, FixedMount
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from tqdm import tqdm

from utils import power_to_daily_energy

class MaxProductionNormalizer:
    def __init__(self):
        self.max_production = None
        self._model_chain = None

    # Create the model chain (from PVLib) with the given system metadata
    def create_model(self, system_metadata):
        latitude = system_metadata['metadata']['loc_latitude']
        longitude = system_metadata['metadata']['loc_longitude']
        altitude = system_metadata['metadata']['loc_altitude']
        Wp_Tot = system_metadata['metadata']['pv_kwp'] * 1000
        loss = system_metadata['metadata']['loss'] * 100

        arrays = []
        for arrayData in system_metadata['arrays']:
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
        self._model_chain = ModelChain(system, location, clearsky_model='ineichen', aoi_model='no_loss', spectral_model="no_loss", losses_model='pvwatts')

    # Generate the estimated maximum daily energy production for the given dates, using the created model chain
    def generate_max_production(self, dates, sampling_freq='1h'):
        if self._model_chain is None:
            raise ValueError("Model not set. Please create a model first.")
        
        # The end date needs to be estimated completly(end date at 23:59). But "endDate" is considered as 00:00 by pd.date_range().
        # So we add 1 day to the end date to include the entire end date in the date_range(), and then we exclude the last value with the inclusive='left' proprety, to remove "endDate+1" at 00:00) in the date_range().
        start_date = dates.min()
        end_date = dates.max() + pd.Timedelta(days=1)

        all_datetimes = pd.date_range(start=start_date, end=end_date, freq=sampling_freq, tz=self._model_chain.location.tz, inclusive='left')

        # Get the clear sky irradiance for the given dates
        weather_clearsky = self._model_chain.location.get_clearsky(all_datetimes)  # In W/m2
        # TODO adjust the clear sky model to take into account the horizon https://pvlib-python.readthedocs.io/en/stable/gallery/shading/plot_simple_irradiance_adjustment_for_horizon_shading.html
        
        # Run the model to get the estimated production
        self._model_chain.run_model(weather_clearsky)
        production_sampling_rate = self._model_chain.results.ac / 1000  # Convert W to kW
        # Convert the power production with a given frequency to the total daily energy
        self.max_production = power_to_daily_energy(production_sampling_rate)
        self.max_production.index = pd.to_datetime(self.max_production.index.date)

    # Tune the physic based model to the given system measures.
    def tune(self, system_measures, window=7):
        if self.max_production is None:
            raise ValueError("Max production not set. Please estimate the max production first.")

        # Remove the obvious outliers. It's important before calculating the std, which can be strongly impacted by the strong outliers.
        outliers_mask = system_measures > 2 * self.max_production[system_measures.index]
        
        valid_system_measures = system_measures[~outliers_mask]
        # if 10% of the data is removed as outliers, we consider that the system is not valid
        if outliers_mask.sum() / system_measures.size > 0.1:
            return None, None, None, None
        # Keep only the max measured value
        # Iterate over windows of a given size, and keep only the maximum value in each window
        max_measured_mask = pd.Series(False, index=system_measures.index)
        for i in range(0, len(system_measures), window):
            window_data = valid_system_measures.iloc[i:i + window]
            if not window_data.empty and not window_data.isna().all():
                max_measured_mask[window_data.idxmax(skipna=True)] = True

        # Calculate the relative difference between the maximum measured and maximum estimated value
        realtive_difference = system_measures[max_measured_mask] / self.max_production

        # Compute statistics
        std = realtive_difference.std()
        mean = realtive_difference.mean()

        # Remove the outilers that have a z-score greater than 1
        z_scores = np.abs(realtive_difference - mean) / std

        # Add the measure with a z-score greater than 1 to the previous outliers (AND operation)
        outliers_mask = outliers_mask | (z_scores > 1)

        # Get the loss that overestimate the estimate maximum daily energy
        loss = 1 - realtive_difference[~outliers_mask].max()

        # Check if the tuning was successful
        unfitted_system = False
        if loss is None or std is None or max_measured_mask is None or outliers_mask is None:
            unfitted_system = True

        # If the std is greater than 1, we remove the system from the list of systems to be processed.
        # This is to avoid to have a system that is not well fitted by the maximum energy estimator model, and that could impact the training of the RF model.
        if std > 1 :
            unfitted_system = True

        return loss, unfitted_system, std, max_measured_mask, outliers_mask

# no data is stored in this class. All data is stored in the data_handler
class MaxProductionNormalizers:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def generate_max_productions(self, tune=True):
        max_productions_dic = {}
        tuner_max_productions_untuned_dic = {}
        tuner_measures_max_mask_dic = {}
        tuner_measures_outliers_mask_dic = {}
        unfitted_systems = []

        for system_name in tqdm(self.data_handler.valid_systems, desc="Normlizer - Generating max productions"):
            system_metadata = self.data_handler.get_metadata(system_name)
            # If we don't want to tune the estimators, we say that the estimator is already tuned
            tuned = not tune

            # reset the loss in the metadata if we want to tune the estimators
            if tune:
                system_metadata['metadata']['loss'] = 0   

            max_production_normalizer = MaxProductionNormalizer()
            while True:  # emulate do while loop
                max_production_normalizer.create_model(system_metadata)
                measures = self.data_handler.get_measures(systems_name=system_name).dropna()
                dates = measures.index
                max_production_normalizer.generate_max_production(dates, sampling_freq='1h')

                # add the estimation to the dictionary
                max_productions_dic[system_name] = max_production_normalizer.max_production

                # Tune estimators
                if tuned:
                    break

                loss, unfitted, std, max_measured_mask, outliers_mask = max_production_normalizer.tune(measures, window=7)

                if unfitted:
                    unfitted_systems.append(system_name)
                    break

                # write the loss in systemsMetadata
                system_metadata['metadata']['loss'] = loss

                # save the untuned estimation to plot the difference before/aftre tuning
                tuner_max_productions_untuned_dic[system_name] = max_production_normalizer.max_production
                tuner_measures_max_mask_dic[system_name] = max_measured_mask
                tuner_measures_outliers_mask_dic[system_name] = outliers_mask

                tuned = True

        # Concatenate all the dictionaries to create dataframe
        self.data_handler.max_production = pd.concat(max_productions_dic, axis=1)
        self.data_handler.tuner_max_productions_untuned = pd.concat(tuner_max_productions_untuned_dic, axis=1)
        self.data_handler.tuner_measures_max_mask = pd.concat(tuner_measures_max_mask_dic, axis=1)
        self.data_handler.tuner_measures_outliers_mask = pd.concat(tuner_measures_outliers_mask_dic, axis=1)

        # remove unfitted_systems from the valid_systems
        self.data_handler.valid_systems = self.data_handler.valid_systems.drop(unfitted_systems)
        for system_name in unfitted_systems:
            print(f"System {system_name} : We can't find the model corresponding to the measured data. This system is removed from the list of systems to be processed.")

        # Save the metadata with the new loss value
        self.data_handler.save_metadata()  

    def normalize(self, data):
        # check that the normalizer has the corresponding value to normalize the data
        self.check_max_production_data_validity(data)
        # Normalize the data
        data_norm = data / self.data_handler.max_production
        return data_norm
    
    def check_max_production_data_validity(self, data):
        # chack that data is either a Series or a DataFrame
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise ValueError("Data must be either a Series or a DataFrame.")
        
        if isinstance(data, pd.Series):
            data = data.to_frame()

        # check that the columns and index in df exist in df2
        if not data.index.isin(self.data_handler.max_production.index).all():
            raise ValueError("Data index is not in the estimated max production. Please calculate the estimated max production for all the observations in the given data.")
        if not data.columns.isin(self.data_handler.max_production.columns).all():
            raise ValueError("Data columns are not in the estimated max production. Please calculate the estimated max production for all the observations in the given data.")      
        
        # check that all the valid input data have a corresponding value in the estimated max production
        mismatch_mask = data.notna() & self.data_handler.max_production.isna()
        if mismatch_mask.any().any():
            raise ValueError("Some values in the estimated max production are null for the given data. Please calculate the estimated max production for all the observations in the given data.")
