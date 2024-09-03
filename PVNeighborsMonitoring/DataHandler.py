import json
import os

import pandas as pd
from tqdm import tqdm
from utils import get_altitude_from_wgs84

class DataHandler:

    def __init__(self, data_dirpath, cache_dirpath):
        self.data_dirpath = data_dirpath
        self.cache_dirpath = cache_dirpath
        self.metadata_filepath = os.path.join(self.data_dirpath, "metadata.json")

        self._measures = None
        self._train_index = None
        self._test_index = None
        self._metadata = None
        
        self.valid_systems = None
        
        self.max_production = None

        self.tuner_max_productions_untuned = None
        self.tuner_measures_max_mask = None
        self.tuner_measures_outliers_mask = None

        self.hsr_outliers_mask = None

    def get_metadata(self, system_name=None):
        if self._metadata is None:
            raise ValueError("Metadata not loaded. Please load the metadata first.")
        if system_name is not None:
            return self._metadata[system_name]
        return self._metadata
    
    def get_measures(self, set='all', systems_name=None):
        if self._measures is None:
            raise ValueError("Measures not loaded. Please load the measures first.")
        if self.valid_systems is None:
            raise ValueError("Valid systems not set. Please check the data integrity first.")
        
        # If systems_name is not set, return all the valid systems
        systems_name = systems_name if systems_name is not None else self.valid_systems

        if set == 'all':
            # take all the observation, for the desired systems, that have at least one value
            measures = self._measures.loc[:, systems_name].dropna(axis='index', how='all').copy()
        elif set == 'train':
            if self._train_index is None:
                raise ValueError("Train index not set. Please create a train-test set first.")
            if self.hsr_outliers_mask is None:
                raise ValueError("Outliers not checked. Please check the outliers first.")
            # take the observation of the training set, for the desired systems, without the outliers values, that have at least one value
            measures = self._measures.loc[self._train_index, systems_name][~self.hsr_outliers_mask].dropna(axis='index', how='all').copy()
        elif set == 'test':
            if self._test_index is None:
                raise ValueError("Test index not set. Please create a train-test set first.")
            # take the observation of the test set, for the desired systems, that have at least one value
            measures = self._measures.loc[self._test_index, systems_name].dropna(axis='index', how='all').copy()
        else:
            raise ValueError("Invalid set value. Please use 'all', 'train' or 'test'.")
        return measures

            
    def get_missing_value(self, sorted=True):
        if sorted:
            # Sort columns by number of missing values
            sorted_columns = self._measures.isnull().sum().sort_values().index
            sorted_measures = self._measures[sorted_columns]

            # Create a boolean DataFrame where True indicates missing values
            missing_values = sorted_measures.isnull()
        else:
            missing_values = self._measures.isnull()
        return missing_values

    def load_metadata(self):
        with open(self.metadata_filepath, 'r') as f:
            self._metadata = json.load(f)

        for _, system_metadata in tqdm(self._metadata.items(), desc="Post-processing metadata"):
            # Add altitude to metadata, if not already present (TODO : imporove with multi threading)
            if "loc_altitude" not in system_metadata['metadata']:
                if "loc_longitude" in system_metadata['metadata'] and "loc_latitude" in system_metadata['metadata']:
                    system_metadata['metadata']["loc_altitude"] = get_altitude_from_wgs84(system_metadata['metadata']["loc_longitude"], system_metadata['metadata']["loc_latitude"])

            # Add the default loss to metadata if not already present
            if 'loss' not in system_metadata['metadata']:
                system_metadata['metadata']['loss'] = 0

            # Convert key with "modX" in the name (x is the array number) to a dictionary with the array number as key
            keys_to_delete = []
            for key, value in system_metadata['metadata'].items():
                if 'mod' in key:
                    # Extract the module number
                    array_num = key.split('_')[1][-1]
                    # Remove the module number from the key
                    new_key = '_'.join(key.split('_')[:1] + key.split('_')[2:])
                    # Add the key-value pair to the appropriate module dictionary
                    if 'arrays' not in system_metadata:
                        system_metadata['arrays'] = {}
                    if array_num not in system_metadata['arrays']:
                        system_metadata['arrays'][array_num] = {}
                    system_metadata['arrays'][array_num][new_key] = value
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del system_metadata['metadata'][key]

        # Save metadata with new format and value
        self.save_metadata()

    def save_metadata(self):
        with open(self.metadata_filepath, 'w') as f:
            json.dump(self._metadata, f, indent=4)

    def load_data(self):
        measures_dic = {}
        duplicates_list = []
        for filename in tqdm(os.listdir(self.data_dirpath), desc="Loading CSV files"):
            if filename.endswith(".csv"):
                system_name = filename.split('_')[0]
                system_measures = pd.read_csv(os.path.join(self.data_dirpath, filename))
                # convert the timestamp to datetime with correct timezone
                system_measures['Datetime'] = pd.to_datetime(system_measures['Timestamp'], unit='ms', utc=True).dt.tz_convert('Europe/Zurich')
                # Convert the datetime to only the date, as the production is the daily production. The +1h is to manage the saving time. Normally PRiOT exports the data at midnight (local time) for the day after (e.g. the energy for the July 1st is saved at July 1st 00:00 Europe/Zurich). However it seams that the saving time is not always correctly handled, and sometime the export is done at 23:00 the day before (e.g. the energy for the July 1st is saved at June 30th 23:00 Europe/Zurich). This is why we add 1h to the datetime to be sure to have the correct date.
                system_measures['Date'] = pd.to_datetime((system_measures['Datetime'] + pd.Timedelta(hours=1)).dt.date)
                # Set the date as index
                system_measures.set_index('Date', inplace=True)
                # Append in duplicates_list all the rows with duplicated index, for logging purpose
                if len(system_measures.index.duplicated(keep=False)):
                    duplicates_list.append(system_measures[system_measures.index.duplicated(keep=False)])
                # keep only the measures tt_forward_active_energy_total_toDay as a Series
                system_measures = system_measures['tt_forward_active_energy_total_toDay']
                # Group by the index (Date) and sum the system_measures for each date to handle duplicates
                system_measures = system_measures.groupby('Date').sum()

                measures_dic[system_name] = system_measures
        # convert the dictionary of series to a pandas dataframe
        self._measures = pd.DataFrame(measures_dic)
        # Log the duplicates
        duplicates_df = pd.concat(duplicates_list)
        log_filename = os.path.join(logs_dirpath, "measureDuplicates.csv")
        print(f"Number of duplicate dates found: {len(duplicates_df)} (see log file {log_filename} for more details)")
        duplicates_df.to_csv(log_filename, index=True)

    def check_integrity(self):
        # Check if the metadata is loaded
        if self._metadata is None:
            raise ValueError("Metadata not loaded. Please load the metadata first.")

        # Check if the measures are loaded
        if self._measures is None:
            raise ValueError("Measures not loaded. Please load the measures first.")

        self.valid_systems = self._measures.columns

        for system_name in tqdm(self._measures.columns, desc="Checking data integrity"):
            valid_system = True

            # Check if the system has measures
            if system_name not in self._measures or self._measures[system_name].count() == 0:
                valid_system = False
                print(f"System {system_name} : No measures found")
            # Check if the system has metadata
            if system_name not in self._metadata:
                valid_system = False
                print(f"System {system_name} : No metadata found")
            else:
                # Check metadata for the entire system
                system_metadata = self._metadata[system_name]
                for key in ['loc_latitude', 'loc_longitude', 'loc_altitude', 'pv_kwp']:
                    # test that the key is present
                    if key not in system_metadata['metadata']:
                        valid_system = False
                        print(f"System {system_name} : No '{key}' found")
                    # if present, convert the value to a number, if possible
                    elif not isinstance(system_metadata['metadata'][key], (int, float)):
                        try:
                            system_metadata['metadata'][key] = int(system_metadata['metadata'][key])
                        except ValueError:
                            try:
                                system_metadata['metadata'][key] = float(system_metadata['metadata'][key])
                            except ValueError:
                                valid_system = False
                                print(f"System {system_name} : The key-value '{key}:{system_metadata['metadata'][key]}' is not a number")

                # Check metadata for the arrays
                if 'arrays' not in system_metadata or len(system_metadata['arrays']) == 0:
                    print(f"System {system_name} : No PV arrays found")
                    valid_system = False
                else:
                    for array_num, array_data in system_metadata['arrays'].items():
                        for key in ['pv_tilt', 'pv_azimut', 'pv_wp', 'pv_number']:
                            if key not in array_data:
                                valid_system = False
                                print(f"System {system_name} : No '{key}' found for array {array_num}")
                            # test that the value is a number, and convert it if possible
                            elif not isinstance(array_data[key], (int, float)):
                                try:
                                    array_data[key] = int(array_data[key])
                                except ValueError:
                                    try:
                                        array_data[key] = float(array_data[key])
                                    except ValueError:
                                        valid_system = False
                                        print(f"System {system_name} : The key-value '{key}:{array_data[key]}' is not a number for array {array_num}")
            if not valid_system:
                self.valid_systems = self.valid_systems.drop(system_name)

        print(f"Number of systems with all the necessary data: {len(self.valid_systems)}/{len(self._measures.columns)}")
    
    def create_train_test_set(self, test_size=None, max_train_size=None, random_state=None, shuffle=True):
        if self.hsr_outliers_mask is None:
            raise ValueError("Outliers not checked. Please check the outliers first.")
        
        self._train_index, self._test_index = train_test_split(self._measures.index, test_size=test_size, random_state=random_state, shuffle=shuffle)
        # remove hsr_outliers_mask from the training set
        if max_train_size is not None and max_train_size < len(self._train_index):
            # Now we want to randomly select "train_size" observation from the training set, with the least number of missing values
            # We will do this by looking at the number of missing values of the "train_size" th element in the sorted list of observation by number of missing values
            # This way, we will know the maximum number of missing values that the selected observation will have
            # Then, we will randomly select "train_size" observation from all the observation with this number of missing value or less

            # Get the training set.            
            training_set = self.get_measures(set='train')
            
            nbr_missing_values_per_day = training_set.isnull().sum(axis=1)
            # Get the maximum number of missing values that the selected observation will have
            max_missing_value = nbr_missing_values_per_day.sort_values().iloc[max_train_size-1]
            # Get all the observation with this number of missing value or less
            valid_observations = training_set[nbr_missing_values_per_day <= max_missing_value]
            # Randomly select "train_size" observation
            self._train_index = valid_observations.sample(n=max_train_size, random_state=random_state).index

    def check_outliers(self, max_threshold=1.1, min_threshold=0.01):
        norm_measures = self.normalize(self.get_measures('all'))
        self.hsr_outliers_mask = (norm_measures > max_threshold) | (norm_measures < min_threshold)
        # If 10% of the values are outliers, remove the system from the list of valid systems
        outliers_count = self.hsr_outliers_mask.sum(axis=0)
        invalid_systems = outliers_count[outliers_count > 0.1 * len(norm_measures)].index
        self.valid_systems = self.valid_systems.drop(invalid_systems)
        for system_name in invalid_systems:
            print(f"System {system_name} : More than 10% of the values are outliers. This system is removed from the list of systems to be trained.")

    def check_training_size(self, min_training_days=14):
        # Calculate the number of valid (non-null) values
        valid_values_count = self.get_measures(set='train').notnull().sum(axis=0)

        # Get the boolean Series where valid values count is less than 14
        invalid_systems = valid_values_count[valid_values_count < min_training_days].index

        # Remove invalid systems from the valid_system list
        self.valid_systems = self.valid_systems.drop(invalid_systems)

        for system_name in invalid_systems:
            print(f"System {system_name} : The system has less than {min_training_days} days of training data. This system is removed from the list of systems to be trained.")
        

        
        